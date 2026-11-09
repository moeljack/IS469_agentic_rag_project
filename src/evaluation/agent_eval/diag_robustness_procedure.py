"""Why do rephrased procedure questions fail? Joins the robustness scores with the question set and the raw run, then
breaks results down by question type and shows, for every failing procedure case, where it went wrong.

Run from the project root, after score_robustness.py:
    python diag_robustness_procedure.py a1_agentic_qwen_robustness
    python diag_robustness_procedure.py a1_agentic_qwen_robustness --qtype procedure --show-trace

Reads (no model is called):
    src/evaluation/eval_questions.json                                       qtype, key facts, gold passages
    src/evaluation/agent_eval/results/robustness/<run>_scores.jsonl           written by score_robustness.py
    src/evaluation/agent_eval/results/robustness/<run>.jsonl                  the raw robustness run (trace, answer)
Adjust the paths below if your layout differs.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "src/evaluation/eval_questions.json"
ROBUST = ROOT / "src/evaluation/agent_eval/results/robustness"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v):
    return "" if v is None else f"{v:.2f}"


def stage_of(p, raw):
    """Where a failing answerable case broke down, in pipeline order."""
    v = p["variant"]
    status = v["status"]
    if status == "error":
        return "crashed"
    if status == "refused":
        return "intake refused it (out of scope)"
    if status == "clarify":
        trace = " ".join((raw or {}).get("trace") or [])
        return "agent asked the user (ask_user)" if "step " in trace else "intake asked for a missing fact"
    if status == "no_answer":
        stop = (raw or {}).get("stop_reason") or ""
        if v.get("hit") == 0 and (v.get("recall") or 0) == 0:
            return f"no_answer, gold never retrieved ({stop})"
        return f"no_answer although gold was retrieved ({stop})"
    # answered but wrong / incomplete
    if (v.get("recall") or 0) < 1:
        return "answered, but part of the gold was not retrieved"
    if v.get("key_fact_hit") is not None and v["key_fact_hit"] < 1:
        return "answered with gold retrieved, key fact missing from answer"
    return "answered (no failure by these metrics)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="robustness run id, e.g. a1_agentic_qwen_robustness")
    ap.add_argument("--qtype", default="procedure")
    ap.add_argument("--show-trace", action="store_true", help="print the trace of each failing case")
    args = ap.parse_args()

    data = {q["qid"]: q for q in json.loads(DATASET.read_text(encoding="utf-8"))}
    scores = read_jsonl(ROBUST / f"{args.run}_scores.jsonl")
    raw_path = ROBUST / f"{args.run}.jsonl"
    raw = {r["case_id"]: r for r in read_jsonl(raw_path)} if raw_path.exists() else {}

    # 1. Is procedure really worse? Same metrics per question type.
    by = defaultdict(list)
    for p in scores:
        if p["answerable"] and p["original"]:
            by[data[p["variant_of"]]["qtype"]].append(p)
    print(f"{'qtype':<12} {'pairs':>5} {'compl_orig':>10} {'compl_var':>9} {'lost':>4} "
          f"{'recall_o':>8} {'recall_v':>8} {'kfh_o':>6} {'kfh_v':>6} {'num_agree':>9} {'no_numbers':>10}")
    for qt, ps in sorted(by.items()):
        lost = sum(1 for p in ps if p["original"]["completed"] >= .5 and p["variant"]["completed"] < .5)
        nonum = sum(1 for p in ps if not data[p["variant_of"]].get("numeric"))
        agree = mean([(p.get("consistency") or {}).get("number_agreement") for p in ps])
        print(f"{qt:<12} {len(ps):>5} {fmt(mean([p['original']['completed'] for p in ps])):>10} "
              f"{fmt(mean([p['variant']['completed'] for p in ps])):>9} {lost:>4} "
              f"{fmt(mean([p['original']['recall'] for p in ps])):>8} {fmt(mean([p['variant']['recall'] for p in ps])):>8} "
              f"{fmt(mean([p['original']['key_fact_hit'] for p in ps])):>6} "
              f"{fmt(mean([p['variant']['key_fact_hit'] for p in ps])):>6} {fmt(agree):>9} {nonum:>10}")
    print("no_numbers = questions marked numeric: false. key_fact_hit and number_agreement only check numbers, so "
          "for these they say little: look at completion and recall instead.\n")

    # 2. Every case of the chosen type: where did it break?
    focus = by.get(args.qtype, [])
    stages = Counter()
    print(f"--- {args.qtype} cases ({len(focus)}) ---")
    for p in sorted(focus, key=lambda p: (p["variant_of"], p["case_id"])):
        r = raw.get(p["case_id"])
        stage = stage_of(p, r)
        stages[stage] += 1
        v, o = p["variant"], p["original"]
        print(f"{p['case_id']:<4} of {p['variant_of']} ({p['kind']}): {o['status']} -> {v['status']} | "
              f"recall {fmt(o['recall'])} -> {fmt(v['recall'])} | {stage}")
        print(f"     Q: {p['question'][:150]}")
        if r:
            route = next((t for t in r.get("trace") or [] if t.startswith("step 1:")), "")
            print(f"     route: {route[:110]}  stop: {r.get('stop_reason')}")
            if v["status"] == "clarify":
                print(f"     asked: {(r.get('answer') or '')[:150]}")
            if args.show_trace and stage != "answered (no failure by these metrics)":
                for t in r.get("trace") or []:
                    print("       |", t[:200])
    print("\nWhere the failures sit:")
    for s, n in stages.most_common():
        print(f"  {n:>3}  {s}")


if __name__ == "__main__":
    main()
