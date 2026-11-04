# Agent evaluation

The agent-specific dimensions of the Eval Plan. The RAG metrics (retrieval, correctness, grounding) live in
`../rag_eval/`; these scripts read the same outputs or run the same systems.

| File | Eval Plan dimension | Needs a model? |
|---|---|---|
| `score_task_success.py` | Task success: RAGAS faithfulness + answer relevance per task, task success rate | Yes (judge, `configs/judge.yaml`) |
| `judge.py` | The judge: config loading, API keys and the LangChain model for RAGAS (copied from rag_eval) | - |
| `configs/judge.yaml` | The judge model (copy of rag_eval's) | - |
| `requirements-scoring.txt` | Environment for `score_task_success.py` (RAGAS, LangChain) | - |
| `IS469_task_success.sh` | Slurm job: starts Ollama, pulls the judge, runs `score_task_success.py` on the GPU node | - |
| `score_trajectory.py` | Step efficiency: wandering/looping rate, step count, budget hits | No |
| `run_suite.py` | Runs the agent (or naive RAG) on the robustness or safety cases | Yes (Ollama, retriever) |
| `injection.py` | Plants a malicious chunk in retrieval results (used by `run_suite.py --suite safety`) | - |
| `score_robustness.py` | Robustness: completion on rephrased questions vs the originals | No |
| `score_safety.py` | Safety / injection resilience: attack success, blocking layer, leaks, hijacked searches, memory writes | No |
| `common.py` | Shared helpers for the scorers | - |
| `data/robustness_cases.json` | 50 rewrites: P01-P05 (paraphrases moved from the dataset) and round 1 (R01-R20) rewrites questions the agent answers; round 2 (R21-R45) is harder: multi-part, previously refused or clarified, and out-of-scope questions (O01-O04, where the right behaviour is to decline). | - |
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
`rag_eval/results/outputs`) with an LLM judge through RAGAS, as the Eval Plan asks:

- **Faithfulness** (RAGAS `Faithfulness`): every claim in the final answer is supported by the chunks the agent
  retrieved. The chunks are the record's `context_used`, exactly what the answer step was shown.
- **Answer relevance** (RAGAS `ResponseRelevancy`): the answer addresses the question asked. Uses the project's
  embedding model (`src.config.EMBED_MODEL_DIR`, or `--embeddings`).
- **Task success** on an answerable question = answered **and** faithfulness >= `--min-faithfulness` (0.8) **and**
  answer relevance >= `--min-relevancy` (0.7). Refusals, clarifications, `no_answer`, errors, empty answers, answers
  without logged chunks and answers the judge could not score all count as failed tasks: they stay in the denominator.
  On an out-of-scope question, success = the agent declined.
- **Task success rate** = share of answerable questions that succeeded, with a bootstrap 95% CI over questions. For a
  config run several times, a question's success is the mean over its repeats (pooled row).

Fix the two thresholds before looking at results; both are written into every output row.

Setup (a separate environment, because RAGAS pins its own LangChain versions):

```
py -3.12 -m venv scoringenv && scoringenv\Scripts\activate
pip install -r src/evaluation/agent_eval/requirements-scoring.txt
```

Set the judge in `configs/judge.yaml` (provider `ollama`, `google`, `anthropic` or `openai`; the key goes in the repo's
`.env`). Then:

```
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --by-type        # all a1 repeats, pooled
python src/evaluation/agent_eval/score_task_success.py --all                            # every agentic run
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen_robustness --suite robustness
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --no-judge       # re-summarise, no judge calls
```

### On the GPU cluster

The judge needs the GPU node, so task success runs as a job: `IS469_task_success.sh` (in this folder) starts Ollama,
pulls the judge named in `configs/judge.yaml` and runs `score_task_success.py` in the scoring environment.
`IS469_eval.sh` does not pull the judge, and `IS469_score.sh` runs only the rag_eval scorers.

One-time setup on the login node, from the project directory:

```
python3 -m venv .venv
.venv/bin/pip install -r src/evaluation/agent_eval/requirements-scoring.txt
```

Then submit from the project directory (`TASK_ARGS` are the scorer's arguments; keep commas out of them):

```
sbatch src/evaluation/agent_eval/IS469_task_success.sh                     # a1_agentic_qwen --by-type --workers 2
sbatch --export=ALL,TASK_ARGS="a1_agentic_qwen_robustness --suite robustness --workers 2" src/evaluation/agent_eval/IS469_task_success.sh
```

Scores are saved every `--batch-size` answers (default 10), so a job stopped by the time limit keeps its work:
resubmit the same command and it carries on. The embedding model runs on the CPU (`--embed-device cpu`) so the judge
has the GPU's memory. Once everything is judged, re-summarise on the login node (no GPU needed):

```
.venv/bin/python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --by-type --no-judge
```

`--suite robustness` scores the rephrased questions from `run_suite.py`: each variant is judged against its own
wording and counted as its original question, so the task success of variants can be compared with the originals.
`--outputs DIR` reads runs from another folder, `--data FILE` uses another question set.

Judge scores are cached in `results/task_success/<run_id>_ragas.csv` per answer and judge model, so a rerun only judges
new answers, and changing the judge model judges everything again rather than mixing judges.

## Outputs

- `agent_eval/results/task_success/`: `<run_id>_ragas.csv` (judge scores), `<run_id>_task_success.jsonl` (one row per
  question: status, scores, success, reason), `task_success_<suite>.{md,csv}` (summary)
- `rag_eval/results/agent_eval/robustness/` and `.../safety/`: `<run_id>.jsonl` (one record per case), `.meta.json`,
  `<run_id>_scores.jsonl`
- `rag_eval/results/scores/<run_id>_trajectory.jsonl`
- `rag_eval/results/report_assets/agent_step_efficiency.*`, `agent_robustness.*`, `agent_safety.*`

Suite outputs are kept out of `results/outputs` on purpose: their case ids are not dataset question ids, so the RAG
scorers' `--all` would fail on them.

## Notes

- Task success imports nothing from `rag_eval`: the code it needs (loading `.env`, the judge model, reading JSONL) is
  copied into `judge.py` and `score_task_success.py`. It only reads the run files rag_eval's `run.py` writes.
- A small local judge often returns output RAGAS cannot parse; those answers show up as `judge_failed` (counted as
  failed tasks). If there are many, use a stronger judge rather than dropping them.
- Safety checks are deterministic string and state tests. Read the answers of the cases marked as succeeded before
  reporting them: an answer that refutes a false rate ("the rate is 17%, not 11.5%") still contains it.
- Planted chunks get a reranker score of 0.97 so they pass the audit cutoff; the test measures the defences after
  retrieval (auditor, answer prompt, scoped actions), not whether a poisoned page would rank.
- The intake gate reads only the user's messages, so a chunk injection cannot write to the user profile by design;
  case I10 checks that this holds.
- `key_fact_hit` (robustness, utility under attack) checks only numeric key facts, without a judge. Use it to compare
  conditions, not as accuracy.