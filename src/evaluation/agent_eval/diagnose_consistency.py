"""Where do inconsistent rephrasing pairs diverge? Reads files only (no model calls).

For every pair score_consistency.py judged inconsistent, compares the original run's record with the rephrased one:
    route      step 1 action (search | decompose | planned), parsed from the trace
    approved   chunk ids the auditor approved over all searches (search_log)
    context    chunk ids shown to the answer step (context_used)
and labels the first stage where the two differ:
    route      the router chose differently, so the searches were different from the start
    retrieval  same route, but the approved / answer-context chunks overlap less than --overlap
    answer     (near) the same evidence, different answer: the answer step is the cause

Run from the project directory, after score_consistency.py has judged the pairs:
    python3 src/evaluation/agent_eval/diagnose_consistency.py a1_agentic_qwen_robustness_rep1 --baseline a1_agentic_qwen
    ... --all-pairs      also show the consistent pairs (useful to see whether they diverge too)
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from src.evaluation.agent_eval.paths import OUTPUT, RESULTS, base_name, read_jsonl  # noqa: E402

STEP1 = re.compile(r"^step 1: (\w+)")


def baseline(name):
    """{qid: first answered record}, same rule as score_consistency.py (run files sorted by run id)."""
    out = {}
    for p in sorted(OUTPUT.glob("*.jsonl")):
        if p.stem == name or base_name(p.stem) == name:
            for rec in read_jsonl(p):
                ok = not rec.get("error") and rec.get("status") == "answered" and (rec.get("answer") or "").strip()
                if ok and rec["qid"] not in out:
                    out[rec["qid"]] = rec
    return out


def route(rec):
    for line in rec.get("trace") or []:
        m = STEP1.match(line)
        if m:
            return m.group(1)
    return None


def approved(rec):
    return {i for s in rec.get("search_log") or [] for i in s.get("approved") or []}


def context(rec):
    return {c["id"] for c in rec.get("context_used") or []}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def stage(orig, var, overlap):
    if route(orig) != route(var):
        return "route"
    if jaccard(approved(orig), approved(var)) < overlap or jaccard(context(orig), context(var)) < overlap:
        return "retrieval"
    return "answer"


def main():
    ap = argparse.ArgumentParser(description="Where inconsistent rephrasing pairs diverge")
    ap.add_argument("run", help="robustness run id, e.g. a1_agentic_qwen_robustness_rep1")
    ap.add_argument("--baseline", required=True, help="normal run of the original questions, e.g. a1_agentic_qwen")
    ap.add_argument("--overlap", type=float, default=0.5, help="chunk overlap (Jaccard) below which retrieval differs")
    ap.add_argument("--all-pairs", action="store_true", help="also list consistent pairs")
    args = ap.parse_args()

    verdicts = {}
    for r in read_jsonl(RESULTS / "consistency" / f"{args.run}_consistency.jsonl"):
        if r.get("consistent") is not None:
            verdicts[r["case_id"]] = r                       # latest successful verdict wins
    if not verdicts:
        raise SystemExit(f"no judged pairs for {args.run}: run score_consistency.py first")
    variants = {r["case_id"]: r for r in read_jsonl(RESULTS / "robustness" / f"{args.run}.jsonl")}
    originals = baseline(args.baseline)

    tally = Counter()
    for cid, v in sorted(verdicts.items(), key=lambda kv: (kv[1]["consistent"], -(kv[1]["contradictions"] or 0))):
        if v["consistent"] and not args.all_pairs:
            continue
        var, orig = variants.get(cid), originals.get(v["variant_of"])
        if var is None or orig is None:
            print(f"{cid}: record missing ({'variant' if var is None else 'original'})\n")
            continue
        where = stage(orig, var, args.overlap)
        tally[(v["consistent"], where)] += 1
        tag = "CONSISTENT" if v["consistent"] else ("CONTRADICTS" if v["contradictions"] else "DIFFERS")
        print(f"=== {cid} (of {v['variant_of']}, {var.get('kind')})  {tag}  -> diverges at: {where}")
        print(f"  original Q: {orig.get('question', '')[:150]}")
        print(f"  variant  Q: {var.get('question', '')[:150]}")
        print(f"  route:      {route(orig)} -> {route(var)}")
        print(f"  stop:       {orig.get('stop_reason')} -> {var.get('stop_reason')}")
        print(f"  queries:    {orig.get('queries')}")
        print(f"          ->  {var.get('queries')}")
        a1, a2, c1, c2 = approved(orig), approved(var), context(orig), context(var)
        print(f"  approved:   {len(a1)} vs {len(a2)} chunks, overlap {jaccard(a1, a2):.0%}")
        print(f"  context:    {len(c1)} vs {len(c2)} chunks, overlap {jaccard(c1, c2):.0%}")
        for d in v.get("differences") or []:
            mark = "CONTRADICTION" if d.get("contradiction") else "difference"
            print(f"  {mark}: {d.get('point', '')[:100]}")
            print(f"      A: {d.get('answer_a', '')[:140]}")
            print(f"      B: {d.get('answer_b', '')[:140]}")
        print()

    print("summary (consistent?, diverges at): count")
    for (ok, where), n in sorted(tally.items()):
        print(f"  {'consistent' if ok else 'inconsistent':<12} {where:<10} {n}")


if __name__ == "__main__":
    main()
