"""Robustness (Eval Plan: "Robustness Score"): does the agent still complete the task when a question is phrased
differently? No model is called.

Each rephrased question (robustness suite, run_suite.py) is compared with its original question, answered in a normal
run (results/outputs, e.g. the three A1 repeats). Per pair:
    completed       status "answered" (refusals, clarification questions, no_answer and errors count as not completed)
    recall / hit    part recall and all-facts-covered of what was retrieved, against the original's gold passages
    key_fact_hit    share of the original's numeric key facts (rates, amounts, dates) found in the answer: a judge-free
                    proxy for correctness, good for comparing the two phrasings, not a substitute for the judge
For an original answered in several repeats, each value is the mean over its repeats.

Summary (results/report_assets/agent_robustness.{md,csv}), by run and by rewrite kind:
    completion_original / completion_variant    the Robustness Score is completion_variant; delta = the drop
    flips_lost / flips_gained                   pairs completed in one phrasing and not the other
    recall, hit, key_fact_hit                   original vs variant

Run:  python src/evaluation/agent_eval/score_robustness.py a1_agentic_qwen_robustness --baseline a1_agentic_qwen
      python src/evaluation/agent_eval/score_robustness.py --all --baseline a1_agentic_qwen
Writes rag_eval/results/agent_eval/robustness/<run_id>_scores.jsonl per run.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.agent_eval.common import (  # noqa: E402
    SUITE_OUT, baseline_records, key_fact_hit, mean, part_recall, rate, save_table, suite_runs)
from src.evaluation.rag_eval.rag_eval_approach.results_io import load_dataset  # noqa: E402


def outcome(rec, q):
    ids = [c["id"] for c in rec.get("retrieved_seen") or []]
    recall, hit = part_recall(ids, q.get("gold_passages") or [])
    status = "error" if rec.get("error") else rec.get("status")
    return {"completed": float(status == "answered"), "status": status, "recall": recall, "hit": hit,
            "key_fact_hit": key_fact_hit(rec.get("answer") if status == "answered" else "", q)}


def averaged(outcomes):
    if not outcomes:
        return None
    keys = ("completed", "recall", "hit", "key_fact_hit")
    out = {k: mean([o[k] for o in outcomes]) for k in keys}
    out["status"] = "/".join(sorted({o["status"] for o in outcomes}))
    return out


def summarise(name, pairs):
    have = [p for p in pairs if p["original"] is not None]
    row = {"name": name, "pairs": len(pairs), "with_original": len(have)}
    if not have:
        return row
    orig_done = [p["original"]["completed"] for p in have]
    var_done = [p["variant"]["completed"] for p in have]
    row.update(
        completion_original=mean(orig_done),
        completion_variant=mean(var_done),
        completion_delta=mean(var_done) - mean(orig_done),
        flips_lost=sum(1 for p in have if p["original"]["completed"] >= 0.5 and p["variant"]["completed"] < 0.5),
        flips_gained=sum(1 for p in have if p["original"]["completed"] < 0.5 and p["variant"]["completed"] >= 0.5),
        recall_original=mean([p["original"]["recall"] for p in have]),
        recall_variant=mean([p["variant"]["recall"] for p in have]),
        hit_original=mean([p["original"]["hit"] for p in have]),
        hit_variant=mean([p["variant"]["hit"] for p in have]),
        key_fact_hit_original=mean([p["original"]["key_fact_hit"] for p in have]),
        key_fact_hit_variant=mean([p["variant"]["key_fact_hit"] for p in have]),
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Robustness to rephrased questions (no LLM)")
    parser.add_argument("runs", nargs="*", help="robustness run ids (a name without _repN takes all repeats)")
    parser.add_argument("--all", action="store_true", help="every run in results/agent_eval/robustness")
    parser.add_argument("--baseline", required=True,
                        help="run in results/outputs that answered the original questions, e.g. a1_agentic_qwen")
    args = parser.parse_args()

    dataset = load_dataset()
    base = baseline_records(args.baseline)
    if not base:
        raise SystemExit(f"no outputs called {args.baseline!r} in results/outputs")
    runs = suite_runs("robustness", args.runs, args.all)
    original = {qid: averaged([outcome(r, dataset[qid]) for r in recs]) for qid, recs in base.items() if qid in dataset}

    table, pooled = [], defaultdict(list)
    for run_id, records in runs.items():
        pairs = []
        for rec in records:
            q = dataset[rec["variant_of"]]
            pairs.append({"case_id": rec["case_id"], "variant_of": rec["variant_of"], "kind": rec.get("kind"),
                          "question": rec["question"], "variant": outcome(rec, q),
                          "original": original.get(rec["variant_of"])})
        (SUITE_OUT / "robustness" / f"{run_id}_scores.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
        table.append(summarise(run_id, pairs))
        for p in pairs:
            pooled["all"].append(p)
            pooled[p["kind"]].append(p)
        missing = sorted({p["variant_of"] for p in pairs if p["original"] is None})
        if missing:
            print(f"note: {run_id}: originals {missing} are not in {args.baseline}; those pairs are left out")
        lost = [p for p in pairs if p["original"] and p["original"]["completed"] >= 0.5 and p["variant"]["completed"] < 0.5]
        if lost:
            print(f"\n{run_id}: rephrasings that lost a completed answer:")
            for p in lost:
                print(f"  {p['case_id']:<4} ({p['kind']}, of {p['variant_of']}) -> {p['variant']['status']}: {p['question'][:70]}")
    if len(runs) > 1:
        table.append(summarise(f"all runs pooled", pooled["all"]))
    for kind in sorted(k for k in pooled if k != "all"):
        table.append(summarise(f"  kind: {kind}", pooled[kind]))
    print()
    save_table("agent_robustness", table,
               extra=f"Originals from `{args.baseline}` (mean over its repeats). key_fact_hit checks numeric key "
                     "facts only, without a judge.")


if __name__ == "__main__":
    main()
