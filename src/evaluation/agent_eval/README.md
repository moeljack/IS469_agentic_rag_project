# Agent evaluation

The agent-specific dimensions of the Eval Plan. The RAG metrics (retrieval, correctness, grounding) live in
`../rag_eval/`. Agent eval keeps its own runs and results: everything it reads and writes is under
`agent_eval/results/` (see `paths.py`). From rag_eval it uses only code (`systems.py` builds the agent and naive RAG)
and the run configs in `rag_eval/rag_eval_approach/configs/`, so both evaluations test the same systems.

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
| `score_cost_latency.py` | Cost and latency (RQ4): tokens and wall-clock seconds per question, agentic vs naive, with paired ratios | No |
| `score_trajectory.py` | Step efficiency: wandering/looping rate, step count, budget hits | No |
| `run_suite.py` | Runs the agent (or naive RAG) on the question set (`--suite main`) or the robustness, safety or tool-call suite | Yes (Ollama, retriever) |
| `injection.py` | Plants a malicious chunk in retrieval results (used by `run_suite.py --suite safety`) | - |
| `score_robustness.py` | Robustness: completion on rephrased questions vs the originals, and judge-free answer consistency (same numbers, same source chunks) | No |
| `score_consistency.py` | Answer consistency: the judge compares each rephrased question's answer with the original's (same answer? contradictions?) | Yes (judge) |
| `prompts/consistency_v1.txt` | The consistency judge prompt | - |
| `score_safety.py` | Safety / injection resilience: attack success, blocking layer, leaks, hijacked searches, memory writes | No |
| `paths.py` | Where agent eval reads and writes (`results/output`, `results/<name>`) | - |
| `common.py` | Shared helpers for the scorers | - |
| `data/robustness_cases.json` | 98 rewrites covering all 80 questions in `eval_questions.json` (see *Robustness cases* below). Round 1 (P01-P05, R01-R20): paraphrases and rewrites of questions the agent answers. Round 2 (R21-R45): harder rewrites of multi-part questions and of the out-of-scope O01-O04. Round 3 (R46-R61): rewrites of multi-step (A), computation (C) and out-of-scope O05-O20 questions. Round 4 (R62-R93): one rewrite of each question the earlier rounds left out, mostly scenario, paraphrase and terse | - |
| `data/injection_cases.json` | 11 chunk injections and 5 user-message injections, each with its success check | - |

## Running

On the GPU machine (the configs are rag_eval's, in `rag_eval/rag_eval_approach/configs/`). First the full runs over the
question set, which the scorers' `--baseline` and the task-success, step-efficiency and cost scorers read:

```
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite main --repeat 1     # also --repeat 2, 3
python src/evaluation/agent_eval/run_suite.py n1_naive.yaml --suite main
python src/evaluation/agent_eval/run_suite.py n2_naive_matched.yaml --suite main
```

They are saved as `results/output/<run_id>[_repN].jsonl` (`a1_agentic_qwen_rep1`, `n1_naive`, ...), with the same
fields as rag_eval's `run.py` records plus the agent's recorded tool calls. Then the suites:

```
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite robustness --repeat 1
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite safety --repeat 1
python src/evaluation/agent_eval/run_suite.py n1_naive.yaml --suite safety        # optional naive comparison
```

Anywhere afterwards (reads files only):

```
python src/evaluation/agent_eval/score_trajectory.py --all
python src/evaluation/agent_eval/score_cost_latency.py --all --by-type
python src/evaluation/agent_eval/score_robustness.py --all --baseline a1_agentic_qwen
python src/evaluation/agent_eval/score_safety.py --all --baseline a1_agentic_qwen
```

`--baseline` is the full run (in `results/output`) whose answers to the original questions the suites are compared
against.

Runs made earlier with rag_eval's `run.py` can be used by copying them in once (see *Moving existing runs* below).

## Robustness cases

`data/robustness_cases.json` holds every rephrased question, one record per case: `case_id`, `variant_of` (the qid of
the original in `eval_questions.json`), `kind` (the rewrite style) and `round`. `run_suite.py --suite robustness` reads
only this file; it checks that every `variant_of` exists in the question set and stops if one does not. The question
set itself holds no paraphrases.

| Round | Cases | Originals | What it tests |
|---|---|---|---|
| 1 | P01-P05, R01-R20 (25) | S01-S07, S09, S11, S12, M02, M03, M08, M10, M13 | Casual, typo and verbose rewrites of questions the agent answers |
| 2 | R21-R45 (25) | M01, M04, M06, M07, M09, S08, S10, S12-S16, M14, M15, O01-O04 | Harder rewrites: multi-part questions, and out-of-scope questions the agent should still decline |
| 3 | R46-R61 (16) | A02-A04, A07, A09, C01, C02, C04, C07, C09, O06, O08, O13, O15, O17, O19 | Multi-step and computation questions, and the off-topic, near-miss and not-in-corpus out-of-scope groups |
| 4 | R62-R93 (32) | S17-S19, S21, M05, M11, M12, M16-M20, A01, A05, A06, A08, A10, C03, C05, C06, C08, C10, O05, O07, O09-O12, O14, O16, O18, O20 | Full coverage: one rewrite of every question rounds 1-3 left out (9 scenario, 8 paraphrase, 7 terse, 5 verbose, 3 typos; scenario, paraphrase, terse and verbose each spread over at least four of the S, M, A, C and O types) |

Rewrite kinds (98 cases in all):

| Kind | Cases | What it changes |
|---|---|---|
| `paraphrase` | 13 | Same meaning in different words, sharing few keywords with the original (P01-P05 are casual paraphrases) |
| `scenario` | 13 | The question wrapped in the user's situation, with details that do not change the answer |
| `terse` | 13 | Keywords and fragments only |
| `verbose` | 9 | The same question with more words and context |
| `colloquial` | 31 | Informal wording, abbreviations, lower case |
| `typos` | 19 | Misspellings of the original's words (figures are never misspelled) |

Computation rewrites keep every figure of the original, so the original's reference answer stays correct.
Out-of-scope rewrites stay out of scope.

Every question has at least one rewrite. Rounds 1 and 2 were written for questions the agent already answered, so on
their own they give an optimistic robustness score; rounds 3 and 4 complete the set, and the "all" row covers every
question. The kinds are not spread evenly over the questions (colloquial is the most common), so compare kinds within
a question type, not across the whole set.

- The baseline must have answered the originals: run the `--suite main` runs on the same `eval_questions.json` as the
  robustness run. Pairs whose original is missing from the baseline are left out, and `score_robustness.py` lists them.
- Add new cases with new ids (R94 onwards) and leave existing ones unchanged, so runs made before and after the change
  still line up case by case. The scorers report each round separately, so earlier results stay comparable on rounds 1
  and 2.
- The scorers read `eval_questions.json` when they run, not when the agent ran. After the gold passages change, re-score
  older runs before comparing them with new ones.

## Task success

Scores the agent's runs over the question set (`results/output`, from `run_suite.py --suite main`) with an LLM judge (LLM-as-a-judge, not BLEU/ROUGE), as the Eval Plan asks. Two judge calls
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

## Cost and latency (RQ4)

`score_cost_latency.py` answers *what does each agentic RAG answer cost compared with naive RAG?* from the full runs
in `results/output/` (agentic and naive configs, `run_suite.py --suite main`). No price is applied:

- **RQ4a, tokens per question:** input + output tokens summed over every LLM call for the question (agent: intake,
  route, decide, planner, auditor, answer; naive: the answer call), with LLM calls per question.
- **RQ4b, wall-clock time per question:** the run's own timer from question to answer (`latency_s.total`).

It reports, per run and pooled over repeats: mean tokens (input, output, total, with a bootstrap 95% CI), median and
p90; mean latency (with a 95% CI), median, p90 and max. For agentic vs naive on the same questions: the token ratio
and latency ratio (agentic mean / naive mean) and the extra tokens and seconds per question, each with a paired
bootstrap 95% CI. It also shows where the agent's time goes (mean seconds per stage), and with `--by-type`, tokens and
time per question type and per status.

```
python3 src/evaluation/agent_eval/score_cost_latency.py --all --by-type
python3 src/evaluation/agent_eval/score_cost_latency.py --all --pair a1_agentic_qwen n1_naive --pair a1_agentic_qwen n2_naive_matched
```

Without `--pair`, every agentic config is compared with every naive one. Questions whose run crashed are left out and
counted. Compare latency only between runs made on the same hardware.

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
Full runs (`--suite main`), robustness and safety runs also record tool calls: score them with `--suite main`,
`--suite robustness` or `--suite safety`.

## Outputs

Everything is under `src/evaluation/agent_eval/results/`:

| Folder | Contents |
|---|---|
| `output/` | Full runs over the question set: `<run_id>[_repN].jsonl` (one record per question) and `.meta.json` (from `run_suite.py --suite main`) |
| `robustness/` | Suite runs `<run_id>_robustness[_repN].jsonl`, `<run_id>_scores.jsonl`, `agent_robustness.{md,csv}` |
| `safety/` | Suite runs `<run_id>_safety[_repN].jsonl`, `<run_id>_scores.jsonl`, `agent_safety.{md,csv}` |
| `toolcalls/` | Suite runs `<run_id>_toolcalls[_<fault>][_repN].jsonl` |
| `trajectory/` | `<run_id>_trajectory.jsonl` (per question), `agent_step_efficiency.{md,csv}` |
| `task_success/` | `<run_id>_judge.csv` (judge scores), `<run_id>_judge_details.jsonl` (claims and question parts), `<run_id>_task_success.jsonl`, `task_success_<suite>.{md,csv}` |
| `tool_calls/` | `<run_id>_tool_calls.jsonl` (per question), `tool_calls_<suite>.{md,csv}` |
| `consistency/` | `<run_id>_consistency.jsonl` (judge verdicts per pair), `consistency.{md,csv}` |
| `cost_latency/` | `cost_latency.{md,csv}` (tables), `<config>_cost_latency.jsonl` (per question) |

### Moving existing runs

Agent eval no longer reads rag_eval's result folders. To use runs made before this change, copy them in once, from the
project directory:

```
mkdir -p src/evaluation/agent_eval/results/output
cp src/evaluation/rag_eval/results/outputs/{a1_agentic_qwen,n1_naive,n2_naive_matched}*.{jsonl,meta.json} src/evaluation/agent_eval/results/output/
for s in robustness safety toolcalls; do
  [ -d src/evaluation/rag_eval/results/agent_eval/$s ] && mkdir -p src/evaluation/agent_eval/results/$s && \
    cp -n src/evaluation/rag_eval/results/agent_eval/$s/* src/evaluation/agent_eval/results/$s/
done
```

Leave out any run that crashed (for example on a GPU PyTorch does not support); rerun it with `run_suite.py` instead.

## Notes

- The scorers import nothing from `rag_eval`. `run_suite.py` imports rag_eval's `systems.py` (and the retrieval and
  model adapters it uses) to build the system under test, and reads the run configs from
  `rag_eval/rag_eval_approach/configs/`; `judge.py` holds copies of the rag_eval helpers it needs.
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
