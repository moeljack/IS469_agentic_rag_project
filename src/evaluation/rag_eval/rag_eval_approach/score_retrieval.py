"""Retrieval metrics for generated runs (no LLM): plan Section 3.1 on what each system actually retrieved.

The retrieved set is every chunk a retrieval step returned (`retrieved_seen`): the top k for the naive systems, the
union over all searches for the agent. Because the agent's set is usually larger, the chunks seen and the chunks used
in the answer prompt are reported beside the metrics; N2's k should be set from A1's "used (mean)".

Run:  python src/evaluation/rag_eval/rag_eval_approach/score_retrieval.py a1_agentic_qwen_rep1 n1_naive
      python src/evaluation/rag_eval/rag_eval_approach/score_retrieval.py --all
Writes results/scores/<run_id>_retrieval.jsonl (recomputed every time; it is cheap).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.evaluation.rag_eval.rag_eval_approach.results_io import OUTPUTS, SCORES, load_dataset, read_jsonl, run_ids
from src.evaluation.rag_eval.rag_eval_retrieval.metrics import mean, part_ranks, percentile, retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description="Retrieval metrics for generated runs")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--all", action="store_true", help="every run in results/outputs")
    args = parser.parse_args()

    dataset = load_dataset()
    SCORES.mkdir(parents=True, exist_ok=True)
    print(f"{'run':<28}{'n':>4}{'Recall':>8}{'Prec':>8}{'F1':>8}{'MRR':>8}{'Hit':>8}{'seen':>7}{'p90':>6}{'used':>7}")
    for run_id in run_ids(args.runs, args.all):
        rows = []
        for rec in read_jsonl(OUTPUTS / f"{run_id}.jsonl"):
            q = dataset[rec["qid"]]
            if not q.get("answerable"):
                continue
            ids = [c["id"] for c in rec.get("retrieved_seen", [])]
            rows.append({"qid": rec["qid"], "n_seen": len(ids), "n_used": len(rec.get("context_used", [])),
                         "part_ranks": part_ranks(ids, q["gold_passages"]),
                         **retrieval_metrics(ids, q["gold_passages"])})
        (SCORES / f"{run_id}_retrieval.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        if not rows:
            print(f"{run_id:<28}   0")
            continue
        m = {k: mean(r[k] for r in rows) for k in ("r", "p", "f1", "rr", "hit", "n_seen", "n_used")}
        print(f"{run_id:<28}{len(rows):>4}{m['r']:>8.0%}{m['p']:>8.0%}{m['f1']:>8.0%}{m['rr']:>8.2f}{m['hit']:>8.0%}"
              f"{m['n_seen']:>7.1f}{percentile([r['n_seen'] for r in rows], 90):>6.0f}{m['n_used']:>7.1f}")
    print("\nseen = chunks retrieved per question (mean, p90); used = chunks in the answer prompt (mean). "
          "For N2, set k to A1's used mean, rounded up.")


if __name__ == "__main__":
    main()
