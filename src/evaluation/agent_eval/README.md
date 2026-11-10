# Agent evaluation

The agent-specific dimensions of the Eval Plan: task success, tool-call accuracy, step efficiency, cost and latency,
robustness, answer consistency and safety. The RAG metrics (retrieval, correctness, grounding) live in `../rag_eval/`.
Run configs come from `rag_eval/rag_eval_approach/configs/`, and all outputs go to `agent_eval/results/`.

## 1. Generate runs (GPU node)

```
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite main --repeat 1     # also --repeat 2, 3
python src/evaluation/agent_eval/run_suite.py n1_naive.yaml --suite main
python src/evaluation/agent_eval/run_suite.py n2_naive_matched.yaml --suite main
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite robustness --repeat 1
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite safety --repeat 1
python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite toolcalls --repeat 1   # optional: --fault empty|error
```

On the cluster, wrap any of these in `sbatch --export=ALL,EVAL_CMD="python3 ..." IS469_eval.sh`.

## 2. Score

```
# Task success: LLM-judged faithfulness and answer relevance, task success rate
python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --by-type

# Tool-call accuracy: schema/argument validity, right tool, tool-step accuracy, error recovery
python3 src/evaluation/agent_eval/score_tool_calls.py --all

# Step efficiency: step count, looping/wandering rate, budget hits
python3 src/evaluation/agent_eval/score_trajectory.py --all

# Cost and latency (RQ4): tokens and seconds per question, agentic vs naive
python3 src/evaluation/agent_eval/score_cost_latency.py --all --by-type

# Robustness: completion and judge-free consistency on rephrased questions
python3 src/evaluation/agent_eval/score_robustness.py --all --baseline a1_agentic_qwen

# Answer consistency: judge-rated same answer / contradiction on rephrased questions
python3 src/evaluation/agent_eval/score_consistency.py a1_agentic_qwen_robustness --baseline a1_agentic_qwen

# Safety: injection attack success, blocking layer, leaks
python3 src/evaluation/agent_eval/score_safety.py --all --baseline a1_agentic_qwen
```

Task success and consistency need the judge model (set in `configs/judge.yaml`), so on the cluster run them as a job:

```
sbatch src/evaluation/agent_eval/IS469_task_success.sh
sbatch --export=ALL,SCORER=score_consistency.py src/evaluation/agent_eval/IS469_task_success.sh
```

Then add `--no-judge` to re-summarise the cached scores on the login node. The other scorers read files only, so they
run anywhere.
