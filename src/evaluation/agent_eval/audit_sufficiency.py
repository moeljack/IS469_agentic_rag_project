"""Did the auditor's verdicts hurt the answers? No model is called.

The auditor approves or rejects each retrieved chunk and says whether the approved chunks are enough (is_sufficient).
When it says enough, the loop stops (stop_reason auditor_sufficient) and the answer is written from what was approved.
This script checks those verdicts against the gold chunks in the question set. For each answerable record:

    gold_parts        facts the question needs (each part = the chunks that state it)
    retrieved_parts   parts with a gold chunk anywhere in what was retrieved (retrieved_seen)
    approved_parts    parts with a gold chunk the auditor approved (search_log[*].approved; context_used if no log)
    context_parts     parts with a gold chunk in the answer's context (context_used)
    key_fact_hit      share of the question's numeric key facts found in the answer (judge-free proxy)
    hedges            sentences where the answer says the documents do not state / mention / cover something

Failure types (an answer can have both):
    stopped_early     auditor_sufficient although some gold part was never approved: the auditor said "enough"
                      while a fact was missing. Split by why it was missing:
                        rejected_gold    the gold chunk WAS retrieved but the auditor rejected it
                        never_retrieved  no search found it, and the auditor did not ask for another search
    rejected_gold     a retrieved gold chunk the auditor rejected (whatever the stop reason)

Run:  python src/evaluation/agent_eval/audit_sufficiency.py a1_agentic_qwen                      (results/output)
      python src/evaluation/agent_eval/audit_sufficiency.py a1_agentic_qwen_robustness --suite robustness
Writes agent_eval/results/auditor/<run_id>_auditor.jsonl and auditor_sufficiency.{md,csv}.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.agent_eval.common import (  # noqa: E402
    OUTPUT, RESULTS, key_fact_hit, load_dataset, mean, rate, read_jsonl, results_dir, save_table)

NOT_RUNS = ("_scores", "_judge", "_judge_details", "_trajectory", "_ragas", "_task_success", "_tool_calls",
            "_consistency", "_cost_latency", "_auditor")
HEDGE = re.compile(r"[^.\n]*\b(?:(?:do|does|did) not (?:explicitly )?(?:specify|state|mention|cover|say|provide|"
                   r"include|address)|not (?:explicitly )?(?:mentioned|specified|stated|covered)|no (?:specific )?"
                   r"information (?:on|about))[^.\n]*", re.I)


def ids_of(items):
    """Chunk ids from a list of ids or of chunk dicts."""
    return {c.get("id") if isinstance(c, dict) else c for c in items or []} - {None}


def covered(ids, gold):
    return [i for i, part in enumerate(gold) if ids & set(part)]


def analyse(rec, q):
    gold = q.get("gold_passages") or []
    log = rec.get("search_log")
    retrieved = ids_of(rec.get("retrieved_seen"))
    context = ids_of(rec.get("context_used"))
    if log is not None:
        approved = set().union(*(ids_of(s.get("approved")) for s in log)) if log else set()
        new = set().union(*(ids_of(s.get("new")) for s in log)) if log else set()
        retrieved |= new
        source = "search_log"
    else:
        approved, source = context, "context_used"          # older runs: the answer's context is the approved set
    status = "error" if rec.get("error") else rec.get("status")
    answer = rec.get("answer") if status == "answered" else ""
    r_parts, a_parts, c_parts = covered(retrieved, gold), covered(approved, gold), covered(context, gold)
    gold_ids = set().union(*map(set, gold)) if gold else set()
    rejected_gold = sorted((retrieved - approved) & gold_ids) if source == "search_log" else []
    missing = [i for i in range(len(gold)) if i not in a_parts]
    sufficient = rec.get("stop_reason") == "auditor_sufficient"
    return {
        "case_id": rec.get("case_id"), "qid": rec.get("qid") or rec.get("variant_of"), "status": status,
        "stop_reason": rec.get("stop_reason"), "source": source,
        "gold_parts": len(gold), "retrieved_parts": len(r_parts), "approved_parts": len(a_parts),
        "context_parts": len(c_parts), "approved_chunks": len(approved), "rejected_gold": rejected_gold,
        "stopped_early": bool(sufficient and gold and missing),
        "missing_retrieved": [i for i in missing if i in r_parts],      # found, then rejected by the auditor
        "missing_never_retrieved": [i for i in missing if i not in r_parts],
        "key_fact_hit": key_fact_hit(answer, q) if answer else None,
        "hedges": [h.strip() for h in HEDGE.findall(answer or "")][:3],
        "question": rec.get("question") or q.get("question"), "answer": (answer or "")[:600],
    }


def summarise(name, rows):
    ans = [r for r in rows if r["gold_parts"] and r["status"] == "answered"]
    suff = [r for r in ans if r["stop_reason"] == "auditor_sufficient"]
    early = [r for r in ans if r["stopped_early"]]
    full = [r for r in suff if not r["stopped_early"]]
    return {
        "name": name, "answered": len(ans), "auditor_sufficient": len(suff),
        "stopped_early": len(early), "stopped_early_rate": rate(len(early), len(suff)),
        "early_gold_rejected": sum(1 for r in early if r["missing_retrieved"]),
        "early_never_retrieved": sum(1 for r in early if r["missing_never_retrieved"] and not r["missing_retrieved"]),
        "any_gold_rejected": sum(1 for r in ans if r["rejected_gold"]),
        "key_fact_hit_full": mean([r["key_fact_hit"] for r in full]),
        "key_fact_hit_early": mean([r["key_fact_hit"] for r in early]),
        "hedging_full": rate(sum(1 for r in full if r["hedges"]), len(full)),
        "hedging_early": rate(sum(1 for r in early if r["hedges"]), len(early)),
    }


def find_runs(folder, names):
    files = {p.stem: p for p in sorted(folder.glob("*.jsonl")) if not p.stem.endswith(NOT_RUNS)}
    picked = {s: p for s, p in files.items() if s in names or re.sub(r"_rep\d+$", "", s) in names}
    if not picked:
        raise SystemExit(f"no runs called {names} in {folder}")
    return picked


def main():
    ap = argparse.ArgumentParser(description="Auditor sufficiency verdicts vs gold chunks (no LLM)")
    ap.add_argument("runs", nargs="+", help="run ids (a name without _repN takes all its repeats)")
    ap.add_argument("--suite", default="output", help="'output' (normal runs) or a suite folder, e.g. robustness")
    ap.add_argument("--show", type=int, default=15, help="early stops to print in full (default 15)")
    args = ap.parse_args()

    dataset = load_dataset()
    folder = OUTPUT if args.suite == "output" else RESULTS / args.suite
    out = results_dir("auditor")
    table, everything = [], []
    for run_id, path in find_runs(folder, args.runs).items():
        rows = []
        for rec in read_jsonl(path):
            q = dataset.get(rec.get("qid") or rec.get("variant_of"))
            if q and q.get("answerable"):
                rows.append(analyse(rec, q))
        (out / f"{run_id}_auditor.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        table.append(summarise(run_id, rows))
        everything += [(run_id, r) for r in rows]
        if any(r["source"] == "context_used" for r in rows):
            print(f"note: {run_id} has no search_log; approved = context_used, so rejected gold can't be seen")
    if len(table) > 1:
        table.append(summarise("pooled", [r for _, r in everything]))

    early = sorted((x for x in everything if x[1]["stopped_early"]),
                   key=lambda x: (x[1]["key_fact_hit"] is None, x[1]["key_fact_hit"] or 0))
    if early:
        print(f"\nAuditor said sufficient while a gold fact was missing ({len(early)}):")
    for run_id, r in early[:args.show]:
        why = []
        if r["missing_retrieved"]:
            why.append(f"parts {r['missing_retrieved']} retrieved but REJECTED")
        if r["missing_never_retrieved"]:
            why.append(f"parts {r['missing_never_retrieved']} never retrieved")
        kfh = "-" if r["key_fact_hit"] is None else f"{r['key_fact_hit']:.0%}"
        print(f"\n  {run_id}  {r['case_id'] or r['qid']}  gold {r['approved_parts']}/{r['gold_parts']} approved "
              f"({r['retrieved_parts']} retrieved), key facts {kfh}: {'; '.join(why)}")
        print(f"    Q: {r['question'][:110]}")
        for h in r["hedges"]:
            print(f"    hedge: {h[:140]}")
        print(f"    A: {r['answer'][:300].replace(chr(10), ' ')}")
    print()
    save_table("auditor_sufficiency", table, folder="auditor",
               extra="stopped_early = the loop stopped on auditor_sufficient although a gold part was never approved. "
                     "early_gold_rejected: the missing gold chunk was retrieved and the auditor rejected it; "
                     "early_never_retrieved: no search found it. key_fact_hit and hedging (answer says the documents "
                     "do not state something) compare early stops with stops that had every gold part.")


if __name__ == "__main__":
    main()
