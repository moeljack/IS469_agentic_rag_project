# RAG evaluation

The RAG dimensions of the Eval Plan: retrieval quality, answer correctness, refusal and grounding, comparing naive
RAG with the agent. The agent-specific metrics live in `../agent_eval/`. Run configs are in
`rag_eval_approach/configs/`, the question set is `../eval_questions.json`, and all outputs go to `rag_eval/results/`.

`run_all.py` runs the four stages below in order and resumes where it stopped, so you can run it again after a crash
or a time limit. To print the commands without running them, use `--dry-run`.

```
python src/evaluation/rag_eval/run_all.py                    # all stages
python src/evaluation/rag_eval/run_all.py --stages 1 2 --skip-api
```

## 1. Retrieval ablation (no LLM)

```
python src/evaluation/rag_eval/rag_eval_retrieval/evaluate_rag.py --mode all
```

Scores R0-R4 (`dense`, `bm25`, `hybrid`, `dense_rerank`, `hybrid_rerank`) against each question's gold chunk ids:
part recall@5, precision@5, F1@5, MRR, hit@5 and median latency. Every later run uses one mode, picked by a rule set
in advance: keep R4 unless its part recall is below R0's, or it is more than 3x slower than R0 with no recall gain.

## 2. Generate answers (GPU node)

```
python src/evaluation/rag_eval/rag_eval_approach/run.py s1_naive_r0_dense.yaml    # naive, dense retrieval
python src/evaluation/rag_eval/rag_eval_approach/run.py n1_naive.yaml             # naive, selected mode
python src/evaluation/rag_eval/rag_eval_approach/run.py a1_agentic_qwen.yaml --repeat 1
python src/evaluation/rag_eval/rag_eval_approach/run.py n2_naive_matched.yaml     # naive, k = A1's mean chunks used
python src/evaluation/rag_eval/rag_eval_approach/run.py g_agentic_openai.yaml     # optional: API generators (g_*.yaml)
```

`run_all.py` sets N2's `k` from the A1 runs for you. Each run writes `results/outputs/<run_id>.jsonl` and a
`.meta.json` that records the config hash, git commit, corpus and machine. A run id refuses to continue if its config
has changed.

On the cluster, `sbatch IS469_eval.sh` runs stages 1 and 2. To run a single command instead, wrap it:
`sbatch --export=ALL,EVAL_CMD="python3 ..." IS469_eval.sh`.

## 3. Score

```
# Retrieval inside each run: gold-chunk recall of the chunks the answer was written from
python src/evaluation/rag_eval/rag_eval_approach/score_retrieval.py --all

# Correctness and refusal: LLM judge against the reference answers
python src/evaluation/rag_eval/rag_eval_approach/score_judge.py --all

# Grounding: faithfulness and context precision
python src/evaluation/rag_eval/rag_eval_approach/score_grounding.py --all

# Optional RAGAS cross-check, in its own environment (requirements-scoring.txt)
python src/evaluation/rag_eval/rag_eval_approach/score_ragas.py --all
```

The judge and grounding scorers need the judge model set in `configs/judge.yaml`. On the cluster, `sbatch IS469_score.sh`
runs the first three scorers with a local Ollama judge, and you can resubmit it to resume.

## 4. Report

```
python src/evaluation/rag_eval/run_all.py --stages 4
```

This rescores whatever is missing, then runs `analyze.py` with the plan's paired comparisons (A1 vs N2, A1 vs N1,
N1 vs S1, and agentic vs naive for each API generator) and a breakdown by question type.

Not automated, by design: choosing the judge, validating it against human labels (`check_judge.py`), and reviewing
the gold passages and reference answers (`db_check/label_gold.py`, `db_check/check_dataset.py`). Do these before you
trust the scores.