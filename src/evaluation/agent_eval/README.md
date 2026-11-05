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
| `IS469_task_success.sh` | Slurm job: starts Ollama, pulls the judge, runs `score_task_success.py` (or `SCORER=score_consistency.py`) on the GPU node | - |
| `score_tool_calls.py` | Tool-call accuracy: schema validity, right tool (route), right arguments (tool-step accuracy), error recovery | No |
| `tool_trace.py` | Records every tool call the agent makes (used by `run_suite.py`), and injects a failed first search for `--fault` | - |
| `score_trajectory.py` | Step efficiency: wandering/looping rate, step count, budget hits | No |
| `run_suite.py` | Runs the agent (or naive RAG) on the robustness, safety or tool-call suite | Yes (Ollama, retriever) |
| `injection.py` | Plants a malicious chunk in retrieval results (used by `run_suite.py --suite safety`) | - |
| `score_robustness.py` | Robustness: completion on rephrased questions vs the originals, and judge-free answer consistency (same numbers, same source chunks) | No |
| `score_consistency.py` | Answer consistency: the judge compares each rephrased question's answer with the original's (same answer? contradictions?) | Yes (judge) |
| `prompts/consistency_v1.txt` | The consistency judge prompt | - |
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

## Answer consistency

The Robustness Score asks whether the agent still answers when a question is rephrased. Consistency asks whether it
gives the same answer. Only answerable pairs where both the original (from `--baseline`) and the rephrased question
were answered are compared.

- **Without a judge**, in `score_robustness.py`'s table: `number_agreement` (overlap of the numbers the two answers
  state; a full date or a YA counts as one value), `number_conflicts` (pairs that state numbers but share none) and
  `context_overlap` (overlap of the chunks each answer was written from). The scorer lists the pairs whose numbers
  differ, with the numbers found only in each answer.
- **With the judge**, `score_consistency.py` shows the judge both answers and asks where they differ, whether any
  difference is a contradiction (both cannot be right), and whether a user would get the same answer from either
  (`prompts/consistency_v1.txt`). It reports `consistency_rate` (same answer, no contradiction),
  `contradiction_rate` and the differing pairs. It runs on the GPU node, like task success:

```
sbatch --export=ALL,SCORER=score_consistency.py src/evaluation/agent_eval/IS469_task_success.sh
sbatch --export=ALL,SCORER=score_consistency.py,TASK_ARGS="a1_agentic_qwen_robustness --baseline a1_agentic_qwen" src/evaluation/agent_eval/IS469_task_success.sh
python3 src/evaluation/agent_eval/score_consistency.py a1_agentic_qwen_robustness --baseline a1_agentic_qwen --no-judge
```

The first two are the same (the second spells out the default). The third re-summarises the cached verdicts on the
login node. Each rephrased answer is compared with the original's first answered repeat, so every pair compares
against the same original. Verdicts are saved every 10 pairs and failed pairs are retried on the next run.

## Tool-call accuracy

The agent's tools are the supervisor's actions: each step the model emits one JSON tool call, the route call at step 1
(search the question as written, or decompose it) and the decide call after that (search with queries, decompose,
run_tool, ask_user, answer, give_up). `tool_trace.py` records every call while the agent runs: the model's raw output,
the queries each step proposed and which the near-duplicate guard dropped, and each executed search (new chunks,
auditor-approved chunks, errors). It wraps the running supervisor from `run_suite.py`, so nothing outside `agent_eval`
changes, and it runs on the existing questions in `eval_questions.json` as they are.

| Metric | Computation |
|---|---|
| Schema validity rate | route, decide and planner calls whose output parsed and matched the tool's JSON schema / calls attempted |
| Argument validity rate | schema-valid route/decide calls whose arguments can work (search has queries, run_tool names an available tool, ask_user has a question) |
| Route agreement (right tool) | step 1 chose decompose for multi-part/multi-step questions and search otherwise (the route prompt's rule, from `qtype`) |
| Tool-step accuracy (right tool + right arguments) | search/decompose steps whose call was valid, ran, and returned evidence the auditor approved / such steps |
| Gold step rate | of those steps, the share that retrieved a gold passage for a fact not yet covered |
| Repeated query rate | queries the model wrote that repeated an earlier search (dropped by the guard) |
| Error recovery rate | errors after which the model's next call was valid, different, ran and returned new chunks / errors |

Errors are malformed calls, invalid arguments, a step whose every query repeated an earlier search, planner failures,
searches that returned nothing new, and retrieval exceptions. The natural ones are rare, so `--fault` injects one into
the first search of every question: `empty` (the search returns nothing) or `error` (it raises). The supervisor does
not catch retrieval exceptions, so expect every `error` case to crash: that is a finding, not a scorer bug.

On the cluster (the agent needs the GPU node, like the other suites):

```
sbatch --export=ALL,EVAL_CMD="python3 src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite toolcalls --repeat 1" IS469_eval.sh
sbatch --export=ALL,EVAL_CMD="python3 src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite toolcalls --fault empty --repeat 1" IS469_eval.sh
sbatch --export=ALL,EVAL_CMD="python3 src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite toolcalls --fault error --repeat 1" IS469_eval.sh
```

Then score on the login node (no model needed):

```
python3 src/evaluation/agent_eval/score_tool_calls.py --all
```

It prints the invalid calls and the errors the agent did not recover from, and writes
`results/tool_calls/<run_id>_tool_calls.jsonl` (per question) and `results/tool_calls/tool_calls_toolcalls.{md,csv}`.
Robustness and safety runs made after this change also record tool calls: score them with `--suite robustness` or
`--suite safety`.

## Outputs

- `agent_eval/results/task_success/`: `<run_id>_judge.csv` (judge scores), `<run_id>_judge_details.jsonl` (claims and
  question parts per answer), `<run_id>_task_success.jsonl` (one row per question: status, scores, success, reason),
  `task_success_<suite>.{md,csv}` (summary)
- `agent_eval/results/consistency/`: `<run_id>_consistency.jsonl` (judge verdicts per pair), `consistency.{md,csv}`
- `agent_eval/results/tool_calls/`: `<run_id>_tool_calls.jsonl` (per question), `tool_calls_<suite>.{md,csv}`
- `agent_eval/results/robustness/`, `.../safety/` and `.../toolcalls/`: `<run_id>.jsonl` (one record per case, from
  `run_suite.py`), `.meta.json`, `<run_id>_scores.jsonl`. Runs made before this move are in
  `rag_eval/results/agent_eval/<suite>/`: the scorers still read them, and `run_suite.py` copies one across when it
  continues it.
- `rag_eval/results/scores/<run_id>_trajectory.jsonl`
- `rag_eval/results/report_assets/agent_step_efficiency.*`, `agent_robustness.*`, `agent_safety.*`

Suite outputs are kept out of `rag_eval/results/outputs` on purpose: their case ids are not dataset question ids, so the RAG
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
