"""Robustness (Eval Plan: "Robustness Score"): does the agent still behave correctly when a question is phrased
differently? No model is called.

Each rephrased question (robustness suite, run_suite.py) is compared with its original question, answered in a normal
run (results/outputs, e.g. the A1 repeats). The right behaviour depends on the original:
    answerable      the agent should answer: `completed` = status "answered" (a refusal, a clarification question,
                    no_answer or an error counts as not completed)
    out-of-scope    the agent should decline: `declined` = status refused, no_answer or clarify (an answer is a failure)
Per answerable pair, also:
    recall / hit    part recall and all-facts-covered of what was retrieved, against the original's gold passages
    key_fact_hit    share of the original's numeric key facts (rates, amounts, dates) found in the answer: a judge-free
                    proxy for correctness, good for comparing the two phrasings, not a substitute for the judge
For an original answered in several repeats, each value is the mean over its repeats.

Errors (a case that raised) are not robustness failures but crashes: they are counted separately and the scorer warns.
Fix the cause and rerun those cases before reporting (move the suite's .jsonl aside, then rerun run_suite.py).

Summary (results/report_assets/agent_robustness.{md,csv}): for the whole run, per round of cases (round 1 = rewrites of
questions the agent answers; round 2 = harder rewrites of multi-part, previously refused and out-of-scope questions)
and per rewrite kind:
    completion_original / completion_variant / completion_delta   the Robustness Score is completion_variant
    flips_lost / flips_gained       answerable pairs completed in one phrasing and not the other
    decline_original / decline_variant / oos_answered               out-of-scope pairs; oos_answered = variants answered
    recall, hit, key_fact_hit       original vs variant (answerable pairs)

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
    SUITE_OUT, baseline_records, key_fact_hit, mean, part_recall, save_table, suite_runs)
from src.evaluation.rag_eval.rag_eval_approach.results_io import load_dataset  # noqa: E402

DECLINED = {"refused", "no_answer", "clarify"}


def outcome(rec, q):
    status = "error" if rec.get("error") else rec.get("status")
    out = {"status": status, "error": float(status == "error"),
           "completed": float(status == "answered"), "declined": float(status in DECLINED)}
    if q.get("answerable"):
        ids = [c["id"] for c in rec.get("retrieved_seen") or []]
        out["recall"], out["hit"] = part_recall(ids, q.get("gold_passages") or [])
        out["key_fact_hit"] = key_fact_hit(rec.get("answer") if status == "answered" else "", q)
    return out


def averaged(outcomes):
    if not outcomes:
        return None
    keys = [k for k in outcomes[0] if k != "status"]
    out = {k: mean([o.get(k) for o in outcomes]) for k in keys}
    out["status"] = "/".join(sorted({o["status"] for o in outcomes}))
    return out


def summarise(name, pairs):
    have = [p for p in pairs if p["original"] is not None and not p["variant"]["error"]]   # crashes are not scored
    ans = [p for p in have if p["answerable"]]
    oos = [p for p in have if not p["answerable"]]
    row = {"name": name, "pairs": len(pairs), "with_original": len(have),
           "errors": int(sum(p["variant"]["error"] for p in pairs))}
    if ans:
        orig = mean([p["original"]["completed"] for p in ans])
        var = mean([p["variant"]["completed"] for p in ans])
        row.update(
            answerable=len(ans), completion_original=orig, completion_variant=var, completion_delta=var - orig,
            flips_lost=sum(1 for p in ans if p["original"]["completed"] >= 0.5 and p["variant"]["completed"] < 0.5),
            flips_gained=sum(1 for p in ans if p["original"]["completed"] < 0.5 and p["variant"]["completed"] >= 0.5),
            recall_original=mean([p["original"]["recall"] for p in ans]),
            recall_variant=mean([p["variant"]["recall"] for p in ans]),
            hit_original=mean([p["original"]["hit"] for p in ans]),
            hit_variant=mean([p["variant"]["hit"] for p in ans]),
            key_fact_hit_original=mean([p["original"]["key_fact_hit"] for p in ans]),
            key_fact_hit_variant=mean([p["variant"]["key_fact_hit"] for p in ans]),
        )
    if oos:
        row.update(out_of_scope=len(oos),
                   decline_original=mean([p["original"]["declined"] for p in oos]),
                   decline_variant=mean([p["variant"]["declined"] for p in oos]),
                   oos_answered=sum(1 for p in oos if p["variant"]["completed"] >= 0.5))
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
                          "round": rec.get("round", 1), "answerable": bool(q.get("answerable")),
                          "question": rec["question"], "variant": outcome(rec, q),
                          "original": original.get(rec["variant_of"])})
        (SUITE_OUT / "robustness" / f"{run_id}_scores.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")
        table.append(summarise(run_id, pairs))
        for p in pairs:
            pooled["all"].append(p)
            pooled[f"round {p['round']}"].append(p)
            pooled[f"kind: {p['kind']}"].append(p)
            pooled["out-of-scope" if not p["answerable"] else "answerable"].append(p)

        errors = [p for p in pairs if p["variant"]["error"]]
        if errors:
            print(f"\nWARNING {run_id}: {len(errors)} case(s) raised an error, so they say nothing about robustness: "
                  f"{', '.join(p['case_id'] for p in errors)}. Fix the cause and rerun them before reporting.")
        missing = sorted({p["variant_of"] for p in pairs if p["original"] is None})
        if missing:
            print(f"note: {run_id}: originals {missing} are not in {args.baseline}; those pairs are left out")
        lost = [p for p in pairs if p["answerable"] and p["original"] and not p["variant"]["error"]
                and p["original"]["completed"] >= 0.5 and p["variant"]["completed"] < 0.5]
        if lost:
            print(f"\n{run_id}: rephrasings that lost a completed answer:")
            for p in lost:
                print(f"  {p['case_id']:<4} ({p['kind']}, of {p['variant_of']}) -> {p['variant']['status']}: "
                      f"{p['question'][:70]}")
        leaked = [p for p in pairs if not p["answerable"] and p["variant"]["completed"] >= 0.5]
        if leaked:
            print(f"\n{run_id}: out-of-scope rephrasings the agent answered (should decline):")
            for p in leaked:
                print(f"  {p['case_id']:<4} ({p['kind']}, of {p['variant_of']}) "
                      f"original {p['original']['status'] if p['original'] else '?'}: {p['question'][:70]}")
    if len(runs) > 1:
        table.append(summarise("all runs pooled", pooled["all"]))
    for key in sorted(k for k in pooled if k.startswith("round ")):
        table.append(summarise(f"  {key}", pooled[key]))
    for key in ("answerable", "out-of-scope"):
        if pooled[key]:
            table.append(summarise(f"  {key}", pooled[key]))
    for key in sorted(k for k in pooled if k.startswith("kind: ")):
        table.append(summarise(f"  {key}", pooled[key]))
    print()
    save_table("agent_robustness", table,
               extra=f"Originals from `{args.baseline}` (mean over its repeats). Answerable pairs: completion = "
                     "answered. Out-of-scope pairs: decline = refused/no_answer/clarify. key_fact_hit checks numeric "
                     "key facts only, without a judge.")


if __name__ == "__main__":
    main()