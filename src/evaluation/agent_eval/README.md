# Agent evaluation

The agent-specific dimensions of the Eval Plan. The RAG metrics (retrieval, correctness, grounding) live in
`../rag_eval/`; these scripts read the same outputs or run the same systems.

| File | Eval Plan dimension | Needs a model? |
|---|---|---|
| `score_task_success.py` | Task success: LLM-judged faithfulness + answer relevance per task, task success rate | Yes (judge, `configs/judge.yaml`) |
| `judge.py` | The judge: config loading, API keys (copied from rag_eval) and the JSON-schema judge call | - |
| `prompts/faithfulness_v1.txt` | Faithfulness judge prompt (copied from rag_eval's `grounding_faithfulness_v1.txt`) | - |
| `prompts/relevancy_v1.txt` | Answer relevance judge prompt | - |
| `configs/judge.yaml` | The judge model (copy of rag_eval's) | - |
| `requirements-scoring.txt` | What `score_task_success.py` needs (ollama, numpy, pandas, pyyaml: already in `.venv`) | - |
| `IS469_task_success.sh` | Slurm job: starts Ollama, pulls the judge, runs `score_task_success.py` on the GPU node | - |
| `score_trajectory.py` | Step efficiency: wandering/looping rate, step count, budget hits | No |
| `run_suite.py` | Runs the agent (or naive RAG) on the robustness or safety cases | Yes (Ollama, retriever) |
| `injection.py` | Plants a malicious chunk in retrieval results (used by `run_suite.py --suite safety`) | - |
| `score_robustness.py` | Robustness: completion on rephrased questions vs the originals | No |
| `score_safety.py` | Safety / injection resilience: attack success, blocking layer, leaks, hijacked searches, memory writes | No |
| `common.py` | Shared helpers for the scorers | - |
| `data/robustness_cases.json` | 45 rewrites: round 1 (R01-R20) rewrites questions the agent answers; round 2 (R21-R45) is harder: multi-part, previously refused or clarified, and out-of-scope questions (O01-O04, where the right behaviour is to decline). The dataset's P01-P05 are added automatically | - |
| `data/injection_cases.json` | 11 chunk injections and 5 user-message injections, each with its success check | - |

## Running

On the GPU machine (same configs as `rag_eval/rag_eval_approach/run.py`):

```
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite robustness --repeat 1
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite safety --repeat 1
python src/evaluation/agent_eval/run_suite.py n1_naive.yaml --suite safety        # optional naive comparison
```

Anywhere afterwards (reads files only):

```
python src/evaluation/agent_eval/score_trajectory.py --all
python src/evaluation/agent_eval/score_robustness.py --all --baseline a1_agentic_qwen
python src/evaluation/agent_eval/score_safety.py --all --baseline a1_agentic_qwen
```

`--baseline` is the normal run (in `rag_eval/results/outputs`) whose answers to the original questions the suites are
compared against.

## Task success

Scores the agent's runs over the question set (made by `rag_eval/rag_eval_approach/run.py`, read from
`rag_eval/results/outputs`) with an LLM judge (LLM-as-a-judge, not BLEU/ROUGE), as the Eval Plan asks. Two judge calls
per answer; each returns JSON whose schema is sent to Ollama, which constrains the reply, so the local judge's output
always parses.

- **Faithfulness**: the share of the answer's factual claims that the chunks the agent retrieved support (RAGAS's
  definition). The chunks are the record's `context_used`, exactly what the answer step was shown. The judge lists the
  claims and marks each supported or not (`prompts/faithfulness_v1.txt`, copied from rag_eval's `score_grounding.py`).
- **Answer relevance**: the share of the parts of the question that the answer addresses. The judge splits the question
  into what it asks for and marks each part addressed or not (`prompts/relevancy_v1.txt`). Saying the information is
  unavailable, or asking the user a question back, does not address a part.
- **Task success** on an answerable question = answered **and** faithfulness >= `--min-faithfulness` (0.8) **and**
  answer relevance >= `--min-relevancy` (0.7). Refusals, clarifications, `no_answer`, errors, empty answers, answers
  without logged chunks and answers the judge could not score all count as failed tasks: they stay in the denominator.
  On an out-of-scope question, success = the agent declined.
- **Task success rate** = share of answerable questions that succeeded, with a bootstrap 95% CI over questions. For a
  config run several times, a question's success is the mean over its repeats (pooled row).

Fix the two thresholds before looking at results; both are written into every output row. With relevance counted by
parts, 0.7 means every part on a question of up to three parts, and all but one on a question of four or more.

RAGAS is not used. Through LangChain it sends the judge no JSON schema, and with the local `gemma4:12b` judge its
faithfulness step failed to parse on almost every answer (rag_eval's plan made the same change for the same reason).
Report the scores as "RAGAS-style LLM-judge scores", and spot-check some answers by reading their claims and parts in
`<run_id>_judge_details.jsonl`.

Set the judge in `configs/judge.yaml` (provider `ollama`, `google`, `anthropic` or `openai`; an API key goes in the
repo's `.env`). Then:

```
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --by-type        # all a1 repeats, pooled
python src/evaluation/agent_eval/score_task_success.py --all                            # every agentic run
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen_robustness --suite robustness
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --no-judge       # re-summarise, no judge calls
```

### On the GPU cluster

The judge needs the GPU node, so task success runs as a job: `IS469_task_success.sh` (in this folder) starts Ollama,
pulls the judge named in `configs/judge.yaml` and runs `score_task_success.py` in the project's `.venv` (nothing extra
to install). `IS469_eval.sh` does not pull the judge, and `IS469_score.sh` runs only the rag_eval scorers.

Submit from the project directory (`TASK_ARGS` are the scorer's arguments; keep commas out of them):

```
sbatch src/evaluation/agent_eval/IS469_task_success.sh                     # a1_agentic_qwen --by-type --workers 1
sbatch --export=ALL,TASK_ARGS="a1_agentic_qwen_robustness --suite robustness" src/evaluation/agent_eval/IS469_task_success.sh
```

Scores are saved every `--batch-size` answers (default 10), so a job stopped by the time limit keeps its work:
resubmit the same command and it carries on. Keep `--workers 1` with the local judge: one Ollama server answers one
request at a time. Once everything is judged, re-summarise on the login node (no GPU needed):

```
python3 src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --by-type --no-judge
```

`--suite robustness` scores the rephrased questions from `run_suite.py`: each variant is judged against its own
wording and counted as its original question, so the task success of variants can be compared with the originals.
`--outputs DIR` reads runs from another folder, `--data FILE` uses another question set.

Judge scores are cached in `results/task_success/<run_id>_judge.csv` per answer, judge model and prompt version, so a
rerun only judges new answers (and answers the judge failed on, up to `--attempts` times), and changing the judge or
a prompt judges everything again rather than mixing scores.

## Outputs

- `agent_eval/results/task_success/`: `<run_id>_judge.csv` (judge scores), `<run_id>_judge_details.jsonl` (claims and
  question parts per answer), `<run_id>_task_success.jsonl` (one row per question: status, scores, success, reason),
  `task_success_<suite>.{md,csv}` (summary)
- `rag_eval/results/agent_eval/robustness/` and `.../safety/`: `<run_id>.jsonl` (one record per case), `.meta.json`,
  `<run_id>_scores.jsonl`
- `rag_eval/results/scores/<run_id>_trajectory.jsonl`
- `rag_eval/results/report_assets/agent_step_efficiency.*`, `agent_robustness.*`, `agent_safety.*`

Suite outputs are kept out of `results/outputs` on purpose: their case ids are not dataset question ids, so the RAG
scorers' `--all` would fail on them.

## Notes

- Task success imports nothing from `rag_eval`: the code it needs (loading `.env`, the judge model, reading JSONL) is
  copied into `judge.py` and `score_task_success.py`. It only reads the run files rag_eval's `run.py` writes.
- An answer the judge could not score shows up as `judge_failed` (counted as a failed task) and is retried on the
  next run, up to `--attempts` times. If there are many, use a stronger judge rather than dropping them.
- Safety checks are deterministic string and state tests. Read the answers of the cases marked as succeeded before
  reporting them: an answer that refutes a false rate ("the rate is 17%, not 11.5%") still contains it.
- Planted chunks get a reranker score of 0.97 so they pass the audit cutoff; the test measures the defences after
  retrieval (auditor, answer prompt, scoped actions), not whether a poisoned page would rank.
- The intake gate reads only the user's messages, so a chunk injection cannot write to the user profile by design;
  case I10 checks that this holds.
- `key_fact_hit` (robustness, utility under attack) checks only numeric key facts, without a judge. Use it to compare
  conditions, not as accuracy.