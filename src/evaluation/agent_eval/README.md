# Agent evaluation

The agent-specific dimensions of the Eval Plan. The RAG metrics (retrieval, correctness, grounding) live in
`../rag_eval/`; these scripts read the same outputs or run the same systems.

| File | Eval Plan dimension | Needs a model? |
|---|---|---|
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

## Outputs

- `rag_eval/results/agent_eval/robustness/` and `.../safety/`: `<run_id>.jsonl` (one record per case), `.meta.json`,
  `<run_id>_scores.jsonl`
- `rag_eval/results/scores/<run_id>_trajectory.jsonl`
- `rag_eval/results/report_assets/agent_step_efficiency.*`, `agent_robustness.*`, `agent_safety.*`

Suite outputs are kept out of `results/outputs` on purpose: their case ids are not dataset question ids, so the RAG
scorers' `--all` would fail on them.

## Notes

- Safety checks are deterministic string and state tests. Read the answers of the cases marked as succeeded before
  reporting them: an answer that refutes a false rate ("the rate is 17%, not 11.5%") still contains it.
- Planted chunks get a reranker score of 0.97 so they pass the audit cutoff; the test measures the defences after
  retrieval (auditor, answer prompt, scoped actions), not whether a poisoned page would rank.
- The intake gate reads only the user's messages, so a chunk injection cannot write to the user profile by design;
  case I10 checks that this holds.
- `key_fact_hit` (robustness, utility under attack) checks only numeric key facts, without a judge. Use it to compare
  conditions, not as accuracy.