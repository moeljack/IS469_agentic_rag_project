"""Tool-call accuracy of the agent (Eval Plan: "Tool-Call Accuracy"): did it use the right tool, with the right
arguments, and recover from errors? No model is called: everything comes from the tool calls run_suite.py recorded
(`tool_calls`, see tool_trace.py).

The agent's tools are the supervisor's actions. Each step the model emits one JSON tool call: at step 1 the route call
(search the question as written | decompose it), after that the decide call (search with queries | decompose |
run_tool | ask_user | answer | give_up). search and decompose run retrieval; the auditor then approves the chunks
that are evidence for the question.

Per question (results/tool_calls/<run_id>_tool_calls.jsonl), and summed over the run:
    calls                 tool calls the model attempted: route + decide calls, plus query-planner calls
    schema_valid          calls whose output parsed as JSON and matched the tool's schema (types, required fields,
                          action in its enum); a planner call is valid unless it failed
                                                                           -> Schema Validity Rate = schema_valid / calls
    argument_valid        schema-valid route/decide calls whose arguments can work for the action (search has
                          queries, run_tool names an available tool, ask_user has a question)
    fallbacks             calls the harness had to replace because the output was unusable
    route_agreement       step 1 chose decompose for a multi-part / multi-step question and search otherwise
                          (the dataset's qtype; the route prompt's own rule) -> "right tool" at the route step
    tool steps            steps where the model's call ran retrieval (route, or a decide search/decompose)
    tool_step_accuracy    tool steps whose call was valid, ran (not dropped as a repeat), and returned evidence the
                          auditor approved -> "right tool with the right arguments" per step
    gold_step_rate        tool steps that retrieved a gold passage for a fact not yet covered (questions with gold)
    repeated_query_rate   queries the model proposed that the near-duplicate guard dropped (it repeated itself)

Error recovery. An error is an observation the agent has to react to:
    malformed_call       the model's output did not parse / match the schema
    invalid_arguments    schema-valid, but the arguments cannot work (see argument_valid)
    repeated_call        every query the model proposed repeated an earlier search, so nothing could run
    planner_error        the query planner failed
    empty_result         a search ran and returned no new chunks (injected by --fault empty, or natural)
    tool_exception       retrieval raised (injected by --fault error, or natural)
For each error, the outcome is read from the model's NEXT decide call:
    recovered            it emitted a valid, different call that ran and returned new chunks
    answered_instead / gave_up / asked_user   it stopped cleanly instead of trying again
    bad_call_again / repeated_again / empty_again   it tried again and failed the same way
    crashed              the question ended with an exception
    loop_ended           no next decide call: a guard or a harness fallback ended the loop
    step_budget          no next decide call: max_steps reached
Error Recovery Rate = recovered / errors (reported separately for injected and natural errors).

Run (anywhere, reads files only):
    python src/evaluation/agent_eval/score_tool_calls.py --all
    python src/evaluation/agent_eval/score_tool_calls.py a1_agentic_qwen_toolcalls a1_agentic_qwen_toolcalls_empty
    python src/evaluation/agent_eval/score_tool_calls.py --all --suite robustness     (runs recorded after this change)
Writes agent_eval/results/tool_calls/<run_id>_tool_calls.jsonl and tool_calls_<suite>.{md,csv}.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
DATASET = EVAL / "eval_questions.json"
SUITES = EVAL / "rag_eval" / "results" / "agent_eval"
OUT = HERE / "results" / "tool_calls"
NOT_RUNS = ("_scores", "_judge", "_judge_details", "_trajectory", "_ragas", "_task_success", "_tool_calls")
MULTI = {"multi-part", "multi-step"}
TOOL_ACTIONS = {"search", "decompose"}
STOPPED = {"answer": "answered_instead", "give_up": "gave_up", "ask_user": "asked_user", "run_tool": "ran_tool"}


# ---------------------------------------------------------------- loading
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def find_runs(folder, names=(), every=False):
    files = sorted(p for p in Path(folder).glob("*.jsonl") if not p.stem.endswith(NOT_RUNS))
    if not every:
        files = [p for p in files if p.stem in names or base_name(p.stem) in names]
        found = {p.stem for p in files} | {base_name(p.stem) for p in files}
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"no outputs for {missing} in {folder} (run run_suite.py first)")
    runs = {p.stem: read_jsonl(p) for p in files}
    runs = {r: recs for r, recs in runs.items() if any(x.get("tool_calls") is not None for x in recs)}
    if not runs:
        raise SystemExit(f"no runs with recorded tool calls in {folder}: run run_suite.py (after the tool_trace.py "
                         "change) first")
    return runs


def question_of(rec, dataset):
    for key in ("qid", "variant_of", "base_qid"):
        if rec.get(key) in dataset:
            return dataset[rec[key]]
    return {}


def expected_route(q):
    """decompose for questions that need several facts, search otherwise (the route prompt's rule); None if unknown."""
    if not q:
        return None
    qtype, kind = str(q.get("qtype") or "").lower(), str(q.get("type") or "").lower()
    if qtype in MULTI or kind.startswith("multi"):
        return "decompose"
    return None if qtype == "out-of-scope" else "search"


# ---------------------------------------------------------------- per question
def _gold_flags(searches, search_log, gold):
    """Per executed search: did it retrieve a gold passage for a fact (gold part) no earlier search covered?
    search events and systems.py's search_log are both one entry per executed retrieval, in order."""
    if not gold or search_log is None or len(search_log) != len(searches):
        return [None] * len(searches)
    covered, flags = set(), []
    for entry in search_log:
        new = set(entry.get("new") or [])
        newly = {i for i, part in enumerate(gold) if i not in covered and set(part) & new}
        covered |= newly
        flags.append(bool(newly))
    return flags


def errors_and_recovery(events, rec):
    """[{type, step, injected, outcome}] for every error the agent had to react to."""
    errors = []
    for i, e in enumerate(events):
        kind = None
        if e["type"] == "decision" and not e["schema_ok"]:
            kind = "malformed_call"
        elif e["type"] == "decision" and e["semantic_issues"]:
            kind = "invalid_arguments"
        elif e["type"] == "queries" and e["none_left"] and e["blocked"] and e["by"] == "model":
            kind = "repeated_call"
        elif e["type"] == "plan" and e["failed"]:
            kind = "planner_error"
        elif e["type"] == "search" and e["error"]:
            kind = "tool_exception"
        elif e["type"] == "search" and not e["new_chunks"]:
            kind = "empty_result"
        if kind:
            errors.append({"type": kind, "step": e.get("step"), "injected": bool(e.get("fault")),
                           "outcome": recovery(events, i, rec)})
    return errors


def recovery(events, i, rec):
    nxt = next((j for j in range(i + 1, len(events))
                if events[j]["type"] == "decision" and events[j]["kind"] == "decide"), None)
    if nxt is None:
        if rec.get("error") or rec.get("status") == "error":
            return "crashed"
        return "step_budget" if rec.get("stop_reason") == "max_steps" else "loop_ended"
    d = events[nxt]
    if not d["schema_ok"] or d["semantic_issues"]:
        return "bad_call_again"
    if d["action"] not in TOOL_ACTIONS:
        return STOPPED.get(d["action"], d["action"] or "unknown")
    later = [e for e in events[nxt + 1:] if e.get("step") == d["step"]]
    q = next((e for e in later if e["type"] == "queries"), None)
    s = next((e for e in later if e["type"] == "search"), None)
    if q is not None and q["none_left"]:
        return "repeated_again"
    if s is None:
        return "crashed" if rec.get("error") else "no_search"
    if s["error"]:
        return "crashed"
    return "recovered" if s["new_chunks"] else "empty_again"


def score_record(rec, q):
    events = rec.get("tool_calls")
    if events is None:
        return None
    decisions = [e for e in events if e["type"] == "decision"]
    plans = [e for e in events if e["type"] == "plan"]
    searches = [e for e in events if e["type"] == "search" and not e["error"]]
    gold_flags = _gold_flags(searches, rec.get("search_log"), q.get("gold_passages") or [])
    gold_by_search = {id(s): g for s, g in zip(searches, gold_flags)}

    steps = []                                   # one per step whose model call chose search/decompose
    for d in decisions:
        if d["action"] not in TOOL_ACTIONS:
            continue
        mine = [e for e in events if e.get("step") == d["step"] and e["type"] in ("queries", "search")]
        qe = next((e for e in mine if e["type"] == "queries"), None)
        se = next((e for e in mine if e["type"] == "search"), None)
        ran = se is not None and not se["error"]
        steps.append({"step": d["step"], "kind": d["kind"], "action": d["action"],
                      "valid": d["schema_ok"] and not d["semantic_issues"],
                      "repeated": bool(qe and qe["none_left"]), "ran": ran,
                      "new_chunks": se["new_chunks"] if ran else 0, "approved": se["approved"] if ran else 0,
                      "gold": gold_by_search.get(id(se)) if ran else (False if q.get("gold_passages") else None)})
    for s in steps:
        s["correct"] = s["valid"] and not s["repeated"] and s["ran"] and s["approved"] > 0

    route = next((d for d in decisions if d["kind"] == "route"), None)
    want = expected_route(q)
    # queries the model wrote itself: the arguments of its decide-step search calls
    model_queries = [e for e in events if e["type"] == "queries" and e["by"] == "model" and e["action"] == "search"
                     and e["step"] > 1]
    proposed = sum(len(e["proposed"]) for e in model_queries)
    blocked = sum(len({x.lower() for x in e["blocked"]} & {x.lower() for x in e["proposed"]}) for e in model_queries)
    errors = errors_and_recovery(events, rec)
    return {
        "item_id": rec.get("case_id") or rec.get("qid"), "qid": q.get("qid"), "qtype": q.get("qtype"),
        "status": "error" if rec.get("error") else rec.get("status"), "stop_reason": rec.get("stop_reason") or "",
        "fault": rec.get("fault"), "reached_loop": bool(decisions),
        "calls": len(decisions) + len(plans),
        "schema_valid": sum(d["schema_ok"] for d in decisions) + sum(not p["failed"] for p in plans),
        "decisions": len(decisions), "argument_valid": sum(d["schema_ok"] and not d["semantic_issues"] for d in decisions),
        "fallbacks": sum(d["fallback"] for d in decisions),
        "route": route["action"] if route else None, "route_expected": want,
        "route_agrees": (route["action"] == want) if route and want else None,
        "tool_steps": len(steps), "correct_steps": sum(s["correct"] for s in steps),
        "steps_with_new_chunks": sum(s["ran"] and s["new_chunks"] > 0 for s in steps),
        "gold_steps": sum(bool(s["gold"]) for s in steps), "gold_known": sum(s["gold"] is not None for s in steps),
        "guard_steps": sum(1 for e in events if e["type"] == "queries" and e["by"] == "guard"),
        "model_queries": proposed, "blocked_queries": blocked,
        "errors": errors, "invalid_calls": [{"step": d["step"], "kind": d["kind"], "schema_errors": d["schema_errors"],
                                             "semantic_issues": d["semantic_issues"], "raw": d["raw"]}
                                            for d in decisions if not d["schema_ok"] or d["semantic_issues"]],
        "steps": steps,
    }


# ---------------------------------------------------------------- summary
def _rate(num, den):
    return num / den if den else None


def summarise(name, rows, runs=1):
    rows = [r for r in rows if r is not None]
    loop = [r for r in rows if r["reached_loop"]]
    total = lambda key: sum(r[key] for r in loop)   # noqa: E731
    routes = [r for r in loop if r["route_agrees"] is not None]
    errors = [e for r in rows for e in r["errors"]]
    injected = [e for e in errors if e["injected"]]
    natural = [e for e in errors if not e["injected"]]
    out = {
        "name": name, "runs": runs, "questions": len(rows), "reached_loop": len(loop),
        "calls": total("calls"), "schema_validity_rate": _rate(total("schema_valid"), total("calls")),
        "argument_validity_rate": _rate(total("argument_valid"), total("decisions")),
        "fallbacks": total("fallbacks"),
        "route_agreement_rate": _rate(sum(r["route_agrees"] for r in routes), len(routes)),
        "route_decompose_rate": _rate(sum(r["route"] == "decompose" for r in loop if r["route"]),
                                      sum(1 for r in loop if r["route"])),
        "tool_steps": total("tool_steps"),
        "tool_step_accuracy": _rate(total("correct_steps"), total("tool_steps")),
        "new_chunk_step_rate": _rate(total("steps_with_new_chunks"), total("tool_steps")),
        "gold_step_rate": _rate(total("gold_steps"), total("gold_known")),
        "guard_steps": total("guard_steps"),
        "model_queries": total("model_queries"),
        "repeated_query_rate": _rate(total("blocked_queries"), total("model_queries")),
        "errors": len(errors),
        "error_recovery_rate": _rate(sum(e["outcome"] == "recovered" for e in errors), len(errors)),
    }
    if injected:
        out["injected_errors"] = len(injected)
        out["injected_recovery_rate"] = _rate(sum(e["outcome"] == "recovered" for e in injected), len(injected))
        out["natural_errors"] = len(natural)
        out["natural_recovery_rate"] = _rate(sum(e["outcome"] == "recovered" for e in natural), len(natural))
    out["error_types"] = ", ".join(f"{k} {v}" for k, v in Counter(e["type"] for e in errors).most_common()) or "-"
    out["recovery_outcomes"] = ", ".join(f"{k} {v}" for k, v in Counter(e["outcome"] for e in errors).most_common()) or "-"
    out["crashes"] = sum(r["status"] == "error" for r in rows)
    return out


def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))

    def cell(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.0%}" if col.endswith(("_rate", "_accuracy")) else f"{v:.2f}".rstrip("0").rstrip(".")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(c, row.get(c)) for c in cols) + " |" for row in table]
    return "\n".join(lines)


def save_table(name, table, note):
    OUT.mkdir(parents=True, exist_ok=True)
    text = markdown(table) + f"\n\n{note}"
    (OUT / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    cols = list(dict.fromkeys(k for row in table for k in row))
    lines = [",".join(cols)] + [",".join(json.dumps(row[c]) if isinstance(row.get(c), str)
                                         else ("" if row.get(c) is None else str(row[c])) for c in cols)
                                for row in table]
    (OUT / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(text)
    print(f"\n  -> {OUT / (name + '.md')}")


def main():
    parser = argparse.ArgumentParser(description="Tool-call accuracy of the agent (no LLM)")
    parser.add_argument("runs", nargs="*", help="run ids (a name without _repN takes all its repeats)")
    parser.add_argument("--all", action="store_true", help="every run with recorded tool calls in the suite folder")
    parser.add_argument("--suite", default="toolcalls", choices=["toolcalls", "robustness", "safety"])
    parser.add_argument("--data", default=str(DATASET), help="question set (default: src/evaluation/eval_questions.json)")
    args = parser.parse_args()
    if not args.runs and not args.all:
        parser.error("name runs to score, or use --all")

    dataset = {q["qid"]: q for q in json.loads(Path(args.data).read_text(encoding="utf-8"))}
    runs = find_runs(SUITES / args.suite, args.runs, args.all)
    OUT.mkdir(parents=True, exist_ok=True)
    table, groups = [], defaultdict(list)
    for run_id, records in runs.items():
        rows = [score_record(rec, question_of(rec, dataset)) for rec in records]
        missing = sum(r is None for r in rows)
        if missing:
            print(f"note {run_id}: {missing} record(s) have no recorded tool calls (made before tool_trace.py), "
                  "left out")
        rows = [r for r in rows if r is not None]
        (OUT / f"{run_id}_tool_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        table.append(summarise(run_id, rows))
        groups[base_name(run_id)].append(rows)
        bad = [(r["item_id"], c) for r in rows for c in r["invalid_calls"]]
        if bad:
            print(f"\n{run_id}: invalid tool calls")
            for item, c in bad[:15]:
                why = "; ".join(c["schema_errors"] + c["semantic_issues"])[:110]
                print(f"  {item:<5} step {c['step']} {c['kind']:<6} {why}")
        failed = [(r["item_id"], e) for r in rows for e in r["errors"] if e["outcome"] != "recovered"]
        if failed:
            print(f"\n{run_id}: errors the agent did not recover from")
            for item, e in failed[:20]:
                tag = " (injected)" if e["injected"] else ""
                print(f"  {item:<5} step {e['step']} {e['type']}{tag} -> {e['outcome']}")
    for base, members in groups.items():
        if len(members) > 1:
            table.append(summarise(f"{base} (pooled {len(members)})", [r for m in members for r in m], len(members)))
    print()
    save_table(f"tool_calls_{args.suite}", table,
               note="Schema validity = model tool calls (route, decide, planner) whose output matched the tool's "
                    "schema. Tool-step accuracy = steps whose search/decompose call was valid, ran, and returned "
                    "auditor-approved evidence. Route agreement = step 1 chose decompose for multi-part/multi-step "
                    "questions and search otherwise. Error recovery = errors after which the next decide call was a "
                    "valid, different call that ran and returned new chunks. Runs ending _empty / _error had a fault "
                    "injected into the first search of every question.")


if __name__ == "__main__":
    main()