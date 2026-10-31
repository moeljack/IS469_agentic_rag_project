"""Step efficiency of the agent (Eval Plan: "Step Efficiency"): did it reach its answer without wandering or looping?

A characteristic failure of agentic RAG is searching again and again without making progress. This scorer turns each
agent run's trajectory into per-question statistics and a run summary. No LLM is called.

Per question (results/scores/<run_id>_trajectory.jsonl):
    steps               actions the supervisor loop took (search, decompose, answer, give_up, ...), 0 if the intake gate
                        refused or clarified before the loop started
    searches            searches actually executed (a search step whose queries were all dropped as near-duplicates is
                        not executed)
    wasted_searches     executed searches that added no new evidence. Evidence = a chunk the auditor approved for the
                        first time; a search whose chunks were all rejected or all fell below the reranker cutoff adds none
    wasted_followups    the same, counting only searches after the first (the agent's own decision to search again)
    redundant_queries   queries that returned no chunk the run had not already seen
    blocked_queries     queries the near-duplicate guard dropped before retrieval (the loop tried to repeat itself)
    budget_hit          stop_reason == max_steps: the loop ran out of steps and answered with what it had
    gold_*              only for runs with a structured `search_log` (systems.py after this change): whether each
                        search retrieved a gold chunk for a fact not yet covered, an objective alternative to the
                        auditor's verdict

Run summary (results/report_assets/agent_step_efficiency.{md,csv}), over questions that reached the loop:
    wasted_search_rate  wasted searches / executed searches         <- the plan's Wandering/Looping Rate
    wasted_followup_rate  wasted follow-up searches / follow-up searches
    steps mean / p90, searches mean, one-shot share (answered after exactly one search), budget-hit rate,
    redundant-query rate, blocked queries, stop-reason counts, LLM calls per question

Run:  python src/evaluation/agent_eval/score_trajectory.py a1_agentic_qwen_rep1 a1_agentic_qwen_rep2
      python src/evaluation/agent_eval/score_trajectory.py --all          (every agentic run in results/outputs)
"""
import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.rag_eval.rag_eval_approach.results_io import (  # noqa: E402
    OUTPUTS, RESULTS, SCORES, load_dataset, read_jsonl, run_ids)

ASSETS = RESULTS / "report_assets"

# Trace line formats written by src/supervisor/supervisor.py
STEP = re.compile(r"^step (\d+): (\w+)")
RETRIEVED = re.compile(r"^\s+retrieved (\d+) chunk\(s\), (\d+) new, for (.+)$")
VERDICT = re.compile(r"^\s+(approved|REJECTED|skipped) ")
BLOCKED = re.compile(r"guard: skipped near-duplicate (?:planned )?queries (\[.*\])\s*$")
AUTO_ANSWER = "skipping the decide call"          # step N: answer chosen by a guard, not by a decide call


# ---------------------------------------------------------------- parsing
def _literal_list(text):
    try:
        value = ast.literal_eval(text)
        return [str(v) for v in value] if isinstance(value, list) else []
    except (ValueError, SyntaxError):
        return []


def steps_from_trace(trace):
    """The loop's steps, in order: [{step, action, auto, queries: [(text, returned, new)], approved, rejected,
    skipped, blocked: [...]}]. Lines before step 1 (profile, intake) are ignored."""
    steps = []
    for line in trace or []:
        m = STEP.match(line)
        if m:
            steps.append({"step": int(m.group(1)), "action": m.group(2), "auto": AUTO_ANSWER in line,
                          "queries": [], "approved": 0, "rejected": 0, "skipped": 0, "blocked": []})
            continue
        if not steps:
            continue
        cur = steps[-1]
        m = RETRIEVED.match(line)
        if m:
            text = m.group(3).strip()
            text = text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text
            cur["queries"].append((text, int(m.group(1)), int(m.group(2))))
            continue
        m = VERDICT.match(line)
        if m:
            cur[{"approved": "approved", "REJECTED": "rejected", "skipped": "skipped"}[m.group(1)]] += 1
            continue
        m = BLOCKED.search(line)
        if m:
            cur["blocked"] += _literal_list(m.group(1))
    return steps


def searches_from_trace(steps):
    """One entry per executed search: {queries, new_chunks, evidence}. Evidence = auditor-approved chunks (every
    audited chunk is new to the run: the retrieval adapter excludes chunks already judged)."""
    return [{"queries": [q for q, _, _ in s["queries"]], "query_new": [n for _, _, n in s["queries"]],
             "new_chunks": sum(n for _, _, n in s["queries"]), "evidence": s["approved"]}
            for s in steps if s["action"] in ("search", "decompose") and s["queries"]]


def searches_from_log(search_log, gold_parts):
    """Same shape from the structured log (systems.py), plus gold coverage: a search adds gold evidence when its new
    chunks cover a fact (gold part) that no earlier search covered."""
    covered, out = set(), []
    for s in search_log:
        new = list(s.get("new") or [])
        newly = {i for i, part in enumerate(gold_parts) if i not in covered and set(part) & set(new)}
        covered |= newly
        out.append({"queries": list(s.get("queries") or []), "query_new": s.get("query_new"),
                    "new_chunks": len(new), "evidence": len(s.get("approved") or []), "gold_parts_added": len(newly)})
    return out, covered


# ---------------------------------------------------------------- scoring
def score_record(rec, q):
    steps = steps_from_trace(rec.get("trace"))
    gold = q.get("gold_passages") or []
    if rec.get("search_log") is not None:
        searches, covered = searches_from_log(rec["search_log"], gold)
        source = "search_log"
    else:
        searches, covered = searches_from_trace(steps), None
        source = "trace"
    queries_new = [n for s in searches for n in (s.get("query_new") or [])]
    wasted = [s["evidence"] == 0 for s in searches]
    row = {
        "qid": rec["qid"], "answerable": bool(q.get("answerable")), "type": q.get("type"), "qtype": q.get("qtype"),
        "status": "error" if rec.get("error") else rec.get("status"), "stop_reason": rec.get("stop_reason") or "",
        "source": source,
        "reached_loop": bool(steps) or bool(searches),
        "steps": max((s["step"] for s in steps), default=0),
        "decide_steps": sum(1 for s in steps if s["step"] > 1 and not s["auto"]),
        "actions": [s["action"] for s in steps],
        "searches": len(searches),
        "followups": max(len(searches) - 1, 0),
        "queries_run": sum(len(s["queries"]) for s in searches),
        "redundant_queries": sum(1 for n in queries_new if n == 0),
        "blocked_queries": sum(len(s["blocked"]) for s in steps),
        "new_chunks": [s["new_chunks"] for s in searches],
        "evidence": [s["evidence"] for s in searches],
        "wasted_searches": sum(wasted),
        "wasted_followups": sum(wasted[1:]),
        "budget_hit": rec.get("stop_reason") == "max_steps",
        "llm_calls": (rec.get("tokens") or {}).get("calls"),
        "latency": (rec.get("latency_s") or {}).get("total", rec.get("wall_s")),
    }
    if covered is not None and gold:
        row["gold_parts_covered"] = len(covered)
        row["gold_parts"] = len(gold)
        row["gold_wasted_searches"] = sum(1 for s in searches if s["gold_parts_added"] == 0)
    return row


def _pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def _rate(num, den):
    return num / den if den else float("nan")


def summarise(name, rows):
    loop = [r for r in rows if r["reached_loop"]]
    ans = [r for r in loop if r["answerable"]]
    searches = sum(r["searches"] for r in loop)
    followups = sum(r["followups"] for r in loop)
    queries = sum(r["queries_run"] for r in loop)
    stops = Counter(r["stop_reason"] or "-" for r in loop)
    out = {
        "name": name,
        "questions": len(rows),
        "stopped_at_intake": len(rows) - len(loop),
        "reached_loop": len(loop),
        "steps_mean": _mean([r["steps"] for r in loop]),
        "steps_median": _pct([r["steps"] for r in loop], 50),
        "steps_p90": _pct([r["steps"] for r in loop], 90),
        "steps_max": max((r["steps"] for r in loop), default=0),
        "searches_mean": _mean([r["searches"] for r in loop]),
        "one_shot_share": _rate(sum(1 for r in loop if r["searches"] == 1 and r["status"] == "answered"), len(loop)),
        "wasted_search_rate": _rate(sum(r["wasted_searches"] for r in loop), searches),
        "wasted_search_rate_answerable": _rate(sum(r["wasted_searches"] for r in ans),
                                               sum(r["searches"] for r in ans)),
        "wasted_followup_rate": _rate(sum(r["wasted_followups"] for r in loop), followups),
        "followup_searches": followups,
        "redundant_query_rate": _rate(sum(r["redundant_queries"] for r in loop), queries),
        "blocked_queries": sum(r["blocked_queries"] for r in loop),
        "budget_hit_rate": _rate(sum(r["budget_hit"] for r in loop), len(loop)),
        "llm_calls_mean": _mean([r["llm_calls"] for r in loop]),
        "stop_reasons": ", ".join(f"{k} {v}" for k, v in stops.most_common()),
    }
    gold = [r for r in loop if "gold_wasted_searches" in r]
    if gold:
        out["gold_wasted_search_rate"] = _rate(sum(r["gold_wasted_searches"] for r in gold),
                                               sum(r["searches"] for r in gold))
    return out


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))
    def cell(col, v):
        if isinstance(v, float):
            if v != v:
                return ""
            if "rate" in col or "share" in col:
                return f"{v:.0%}"
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(c, row.get(c, "")) for c in cols) + " |" for row in table]
    return "\n".join(lines)


def is_agentic(run_id):
    recs = read_jsonl(OUTPUTS / f"{run_id}.jsonl")
    return bool(recs) and any(r.get("system") == "agentic" for r in recs)


def main():
    parser = argparse.ArgumentParser(description="Step efficiency of agentic runs (no LLM)")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--all", action="store_true", help="every agentic run in results/outputs")
    parser.add_argument("--loopers", type=int, default=2,
                        help="list questions with at least this many wasted follow-up searches (default 2)")
    args = parser.parse_args()

    dataset = load_dataset()
    SCORES.mkdir(parents=True, exist_ok=True)
    ids = [r for r in run_ids(args.runs, args.all) if is_agentic(r)]
    skipped = sorted(set(run_ids(args.runs, args.all)) - set(ids))
    if skipped:
        print(f"skipping non-agentic runs (one search per question by design): {skipped}")
    if not ids:
        raise SystemExit("no agentic runs to score")

    per_run, table = {}, []
    for run_id in ids:
        rows = [score_record(rec, dataset[rec["qid"]]) for rec in read_jsonl(OUTPUTS / f"{run_id}.jsonl")
                if rec["qid"] in dataset]
        per_run[run_id] = rows
        (SCORES / f"{run_id}_trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                                           encoding="utf-8")
        table.append(summarise(run_id, rows))
    groups = {}
    for run_id in ids:
        groups.setdefault(base_name(run_id), []).append(run_id)
    for name, members in groups.items():
        if len(members) > 1:
            table.append(summarise(f"{name} (pooled {len(members)})", [r for m in members for r in per_run[m]]))

    ASSETS.mkdir(parents=True, exist_ok=True)
    text = markdown(table)
    (ASSETS / "agent_step_efficiency.md").write_text(text + "\n", encoding="utf-8")
    cols = list(dict.fromkeys(k for row in table for k in row))
    csv = [",".join(cols)] + [",".join(json.dumps(row.get(c, "")) if isinstance(row.get(c), str) else
                                       ("" if row.get(c) is None else str(row.get(c))) for c in cols) for row in table]
    (ASSETS / "agent_step_efficiency.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(text)
    print(f"\n  -> {ASSETS / 'agent_step_efficiency.md'}")

    loopers = [(run_id, r) for run_id, rows in per_run.items() for r in rows
               if r["wasted_followups"] >= args.loopers or r["budget_hit"]]
    if loopers:
        print(f"\nWandering cases (>= {args.loopers} wasted follow-ups, or step budget hit):")
        for run_id, r in loopers:
            print(f"  {run_id:<26} {r['qid']:<4} {'answerable' if r['answerable'] else 'out-of-scope':<12} "
                  f"steps {r['steps']} searches {r['searches']} evidence/search {r['evidence']} "
                  f"blocked {r['blocked_queries']} stop {r['stop_reason']} -> {r['status']}")
    sources = Counter(r["source"] for rows in per_run.values() for r in rows if r["reached_loop"])
    if sources.get("trace"):
        print(f"\nnote: {sources['trace']} question(s) were scored from trace text (runs made before search_log was "
              "logged); gold-based evidence needs a rerun.")


if __name__ == "__main__":
    main()
