"""Robustness (Eval Plan: "Robustness Score"): does the agent still behave correctly when a question is phrased
differently, and does it give the same answer? No model is called.

Each rephrased question (robustness suite, run_suite.py) is compared with its original question, answered in a normal
run (agent_eval/results/output, e.g. the A1 repeats). The right behaviour depends on the original:
    answerable      the agent should answer: `completed` = status "answered" (a refusal, a clarification question,
                    no_answer or an error counts as not completed)
    out-of-scope    the agent should decline: `declined` = status refused, no_answer or clarify (an answer is a failure)
Per answerable pair, also:
    recall / hit    part recall and all-facts-covered of what was retrieved, against the original's gold passages
    key_fact_hit    share of the original's numeric key facts (rates, amounts, dates) found in the answer: a judge-free
                    proxy for correctness, good for comparing the two phrasings, not a substitute for the judge
For an original answered in several repeats, each value is the mean over its repeats.

Answer consistency (answerable pairs where both the rephrased question and the original were answered): is the answer
to the rephrased question the same as the answer to the original? Judge-free checks here; score_consistency.py asks the
judge to compare the two answers in full.
    number_agreement   overlap (Jaccard) of the numbers the two answers state: rates, amounts, dates, counts.
                       Citation markers ([Doc 1]) are ignored. 1 = the same numbers; None when neither states one
    number_conflicts   pairs where both answers state numbers and share none: the clearest sign they disagree
    context_overlap    overlap (Jaccard) of the chunks each answer was written from (`context_used`)
Against an original answered in several repeats, each value is the mean over those repeats.

Errors (a case that raised) are not robustness failures but crashes: they are counted separately and the scorer warns.
Fix the cause and rerun those cases before reporting (move the suite's .jsonl aside, then rerun run_suite.py).

Summary (agent_eval/results/robustness/agent_robustness.{md,csv}): for the whole run, per round of cases (round 1 =
rewrites of questions the agent answers; round 2 = harder rewrites of multi-part, previously refused and out-of-scope
questions) and per rewrite kind:
    completion_original / completion_variant / completion_delta   the Robustness Score is completion_variant
    flips_lost / flips_gained       answerable pairs completed in one phrasing and not the other
    decline_original / decline_variant / oos_answered               out-of-scope pairs; oos_answered = variants answered
    recall, hit, key_fact_hit       original vs variant (answerable pairs)
    both_answered, number_agreement, number_conflicts, context_overlap   answer consistency

Run:  python src/evaluation/agent_eval/score_robustness.py a1_agentic_qwen_robustness --baseline a1_agentic_qwen
      python src/evaluation/agent_eval/score_robustness.py --all --baseline a1_agentic_qwen
Writes agent_eval/results/robustness/<run_id>_scores.jsonl per run.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.agent_eval.common import (  # noqa: E402
    SUITE_OUT, baseline_records, key_fact_hit, load_dataset, mean, norm, part_recall, save_table, suite_runs)

DECLINED = {"refused", "no_answer", "clarify"}
CITATION = re.compile(r"\[doc \d+\]")
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE = re.compile(rf"\b\d{{1,2}} {_MON},? \d{{4}}\b|\b{_MON} \d{{1,2}},? \d{{4}}\b|\bya ?\d{{4}}\b")   # one value each
NUMBER = re.compile(r"\$?\d+(?:\.\d+)?%?")


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


# ---------------------------------------------------------------- answer consistency (judge-free)
def numbers(answer):
    """The numbers an answer states, normalised as norm() does (S$ -> $, no thousands commas, short month names),
    citations removed. A full date ("30 nov 2026") or a YA ("ya 2026") counts as one value, so two answers whose
    deadlines differ only in the year do not share the day."""
    text = CITATION.sub(" ", norm(answer))
    dates = set()
    for d in DATE.findall(text):
        parts = re.sub(r"[ ,]+", " ", d).split()
        if parts[0] == "ya" or d.startswith("ya"):
            dates.add("ya" + re.sub(r"\D", "", d))
        elif parts[0].isdigit():                              # 30 nov 2026
            dates.add(f"{int(parts[0])} {parts[1][:3]} {parts[2]}")
        else:                                                 # nov 30 2026 -> 30 nov 2026
            dates.add(f"{int(parts[1])} {parts[0][:3]} {parts[2]}")
    text = DATE.sub(" ", text)
    return dates | {n.rstrip(".") for n in NUMBER.findall(text)}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def answered(rec):
    return not rec.get("error") and rec.get("status") == "answered" and bool((rec.get("answer") or "").strip())


def consistency(rec, originals):
    """Compare the rephrased question's answer with each answered repeat of the original (None if not comparable)."""
    others = [o for o in originals or [] if answered(o)]
    if not answered(rec) or not others:
        return None
    mine, ctx = numbers(rec["answer"]), {c["id"] for c in rec.get("context_used") or []}
    agree, conflict, overlap = [], [], []
    for o in others:
        theirs = numbers(o["answer"])
        agree.append(jaccard(mine, theirs))
        conflict.append(float(bool(mine and theirs and not mine & theirs)))
        overlap.append(jaccard(ctx, {c["id"] for c in o.get("context_used") or []}))
    first = numbers(others[0]["answer"])
    return {"compared": len(others), "number_agreement": mean(agree), "number_conflict": mean(conflict),
            "context_overlap": mean(overlap),
            "only_variant": sorted(mine - first), "only_original": sorted(first - mine)}


def summarise(name, pairs):
    have = [p for p in pairs if p["original"] is not None and not p["variant"]["error"]]   # crashes are not scored
    ans = [p for p in have if p["answerable"]]
    oos = [p for p in have if not p["answerable"]]
    row = {"name": name, "pairs": len(pairs), "with_original": len(have),
           "errors": int(sum(p["variant"]["error"] for p in pairs))}
    if ans:
        orig = mean([p["original"]["completed"] for p in ans])
        var = mean([p["variant"]["completed"] for p in ans])
        both = [p for p in ans if p.get("consistency")]
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
            both_answered=len(both),
            number_agreement=mean([p["consistency"]["number_agreement"] for p in both]),
            number_conflicts=sum(1 for p in both if p["consistency"]["number_conflict"] >= 0.5),
            context_overlap=mean([p["consistency"]["context_overlap"] for p in both]),
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
    parser.add_argument("--all", action="store_true", help="every run in agent_eval/results/robustness")
    parser.add_argument("--baseline", required=True,
                        help="run in agent_eval/results/output that answered the original questions, e.g. a1_agentic_qwen")
    parser.add_argument("--differing", type=float, default=0.5,
                        help="list answerable pairs whose number_agreement is below this (default 0.5)")
    args = parser.parse_args()

    dataset = load_dataset()
    base = baseline_records(args.baseline)
    if not base:
        raise SystemExit(f"no outputs called {args.baseline!r} in agent_eval/results/output "
                         "(run run_suite.py --suite main, or copy the runs there)")
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
                          "original": original.get(rec["variant_of"]),
                          "consistency": consistency(rec, base.get(rec["variant_of"])) if q.get("answerable") else None})
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
        differ = [p for p in pairs if p.get("consistency") and p["consistency"]["number_agreement"] is not None
                  and p["consistency"]["number_agreement"] < args.differing]
        if differ:
            print(f"\n{run_id}: answers whose numbers differ from the original's (number_agreement < {args.differing}):")
            for p in sorted(differ, key=lambda p: p["consistency"]["number_agreement"]):
                c = p["consistency"]
                print(f"  {p['case_id']:<4} (of {p['variant_of']}) agreement {c['number_agreement']:.2f}"
                      f"{'  CONFLICT' if c['number_conflict'] >= 0.5 else ''}  only in rephrased: "
                      f"{c['only_variant'][:6]}  only in original: {c['only_original'][:6]}")
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
    save_table("agent_robustness", table, folder="robustness",
               extra=f"Originals from `{args.baseline}` (mean over its repeats). Answerable pairs: completion = "
                     "answered. Out-of-scope pairs: decline = refused/no_answer/clarify. key_fact_hit checks numeric "
                     "key facts only, without a judge. Consistency (pairs where both versions were answered): "
                     "number_agreement = overlap of the numbers the two answers state, number_conflicts = pairs that "
                     "share none, context_overlap = overlap of the chunks they were written from. For a full "
                     "comparison of the two answers, run score_consistency.py.")


if __name__ == "__main__":
    main()