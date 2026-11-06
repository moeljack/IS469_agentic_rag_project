"""Answer consistency under rephrasing (Eval Plan: "Robustness"): does the agent give the same answer when a question
is asked in different words? The LLM judge compares the two answers in full.

score_robustness.py checks whether the agent still answers a rephrased question and, without a judge, whether the two
answers state the same numbers. This script shows the judge both answers, the one to the original question (from a
normal run in agent_eval/results/output, --baseline) and the one to the rephrased question (robustness suite, run_suite.py), and asks where they
differ (prompts/consistency_v1.txt; one call per pair, JSON constrained by a schema like the task-success judge):

    differences    each point where the answers differ: what A says, what B says, and whether they contradict each
                   other (cannot both be right) or one only adds or leaves out a detail
    same_answer    the judge's verdict: a user would come away with the same answer from either one

Per pair, consistent = same_answer and no contradiction. Only answerable pairs where both versions were answered are
compared (a refusal or clarification has nothing to compare; score_robustness.py already counts those). The original
answer is the first answered repeat of the baseline (sorted by run id), so every pair compares the same original.

Summary: consistency_rate (consistent / judged), contradiction_rate (pairs with at least one contradiction / judged),
same_answer_rate, mean differences per pair, per round and per rewrite kind. Judge failures count in judge_failed and
are retried on the next run, up to --attempts times.

Run on the GPU node (the judge is local Ollama), from the project directory:
    sbatch --export=ALL,SCORER=score_consistency.py,TASK_ARGS="a1_agentic_qwen_robustness --baseline a1_agentic_qwen" \
        src/evaluation/agent_eval/IS469_task_success.sh
Re-summarise anywhere without judge calls:
    python3 src/evaluation/agent_eval/score_consistency.py a1_agentic_qwen_robustness --baseline a1_agentic_qwen --no-judge
Writes agent_eval/results/consistency/<run_id>_consistency.jsonl (cached judge verdicts) and consistency.{md,csv}.
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.agent_eval.judge import DEFAULT_JUDGE, json_judge, judge_name, load_judge, read_judge  # noqa: E402
from src.evaluation.agent_eval.paths import DATASET, OUTPUT, RESULTS  # noqa: E402

BASELINES = OUTPUT                                  # agent_eval/results/output/: the original questions' runs
SUITE_DIRS = [RESULTS / "robustness"]               # agent_eval/results/robustness/: the rephrased questions' runs
OUT = RESULTS / "consistency"
PROMPT = HERE / "prompts" / "consistency_v1.txt"
METHOD = "consistency-v1"
NOT_RUNS = ("_scores", "_judge", "_judge_details", "_trajectory", "_ragas", "_task_success", "_tool_calls",
            "_consistency")
SCHEMA = {"type": "object", "required": ["differences", "same_answer"], "properties": {
    "differences": {"type": "array", "items": {"type": "object", "required": ["point", "answer_a", "answer_b",
                                                                               "contradiction"],
                                               "properties": {"point": {"type": "string"},
                                                              "answer_a": {"type": "string"},
                                                              "answer_b": {"type": "string"},
                                                              "contradiction": {"type": "boolean"}}}},
    "same_answer": {"type": "boolean"}}}


# ---------------------------------------------------------------- loading
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def find_runs(names=(), every=False):
    by_stem = {}
    for d in SUITE_DIRS:
        by_stem.update({p.stem: p for p in d.glob("*.jsonl") if not p.stem.endswith(NOT_RUNS)})
    stems = sorted(by_stem) if every else [s for s in sorted(by_stem) if s in names or base_name(s) in names]
    if not stems:
        raise SystemExit(f"no robustness outputs for {list(names) or 'any run'} in {' or '.join(map(str, SUITE_DIRS))}")
    return {s: read_jsonl(by_stem[s]) for s in stems}


def baseline(name):
    """{qid: the first answered record}, over the baseline's run files sorted by run id."""
    out = {}
    for p in sorted(BASELINES.glob("*.jsonl")):
        if p.stem == name or base_name(p.stem) == name:
            for rec in read_jsonl(p):
                if answered(rec) and rec["qid"] not in out:
                    out[rec["qid"]] = dict(rec, _run=p.stem)
    return out


def answered(rec):
    return not rec.get("error") and rec.get("status") == "answered" and bool((rec.get("answer") or "").strip())


# ---------------------------------------------------------------- judging
def parse(raw):
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("differences"), list) \
                and isinstance(data.get("same_answer"), bool):
            data["differences"] = [d for d in data["differences"] if isinstance(d, dict)]
            return data
    return None


def judge_pair(call, template, pair, tries=2):
    prompt = (template.replace("{question}", pair["question"]).replace("{variant_question}", pair["variant_question"])
              .replace("{answer_a}", pair["answer_a"]).replace("{answer_b}", pair["answer_b"]))
    raw = ""
    for _ in range(tries):
        try:
            raw = call(prompt, SCHEMA)
        except Exception as e:                       # timeout or server error: try again, then give up
            raw = f"ERROR {type(e).__name__}: {e}"
            continue
        verdict = parse(raw)
        if verdict is not None:
            diffs = verdict["differences"]
            contradictions = [d for d in diffs if d.get("contradiction")]
            return {"same_answer": verdict["same_answer"], "differences": diffs, "n_differences": len(diffs),
                    "contradictions": len(contradictions),
                    "consistent": verdict["same_answer"] and not contradictions, "raw": None}
    return {"same_answer": None, "differences": None, "n_differences": None, "contradictions": None,
            "consistent": None, "raw": raw[:2000]}


def pairs_for(records, originals, dataset):
    """Comparable pairs: answerable, rephrased version answered, original answered in the baseline."""
    out, skipped = [], Counter()
    for rec in records:
        q = dataset.get(rec.get("variant_of"))
        if not q or not q.get("answerable"):
            skipped["out_of_scope"] += 1
            continue
        if not answered(rec):
            skipped[f"rephrased_{'error' if rec.get('error') else rec.get('status')}"] += 1
            continue
        orig = originals.get(rec["variant_of"])
        if orig is None:
            skipped["original_not_answered"] += 1
            continue
        out.append({"case_id": rec["case_id"], "variant_of": rec["variant_of"], "kind": rec.get("kind"),
                    "round": rec.get("round", 1), "question": q["question"], "variant_question": rec["question"],
                    "answer_a": orig["answer"].strip(), "answer_b": rec["answer"].strip(), "original_run": orig["_run"]})
    return out, skipped


def judge_run(run_id, pairs, cfg, call, template, workers, batch_size, attempts):
    path, key = OUT / f"{run_id}_consistency.jsonl", f"{judge_name(cfg)}|{METHOD}"
    rows = [r for r in read_jsonl(path) if r.get("judge") == key]
    tries = Counter(r["case_id"] for r in rows)
    judged = {r["case_id"] for r in rows if r.get("consistent") is not None}
    todo = [p for p in pairs if p["case_id"] not in judged and tries[p["case_id"]] < attempts]
    gave_up = sum(1 for p in pairs if p["case_id"] not in judged and tries[p["case_id"]] >= attempts)
    print(f"{run_id}: {len(judged)} pairs already judged by {key}, {gave_up} failed {attempts} times, "
          f"{len(todo)} to judge")
    for start in range(0, len(todo), max(1, batch_size)):
        batch, t0 = todo[start:start + max(1, batch_size)], time.perf_counter()
        one = lambda p: judge_pair(call, template, p)   # noqa: E731
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = list(pool.map(one, batch))
        else:
            verdicts = [one(p) for p in batch]
        with path.open("a", encoding="utf-8") as out:
            for p, v in zip(batch, verdicts):
                out.write(json.dumps({"case_id": p["case_id"], "variant_of": p["variant_of"], "judge": key,
                                      "original_run": p["original_run"], **v}, ensure_ascii=False) + "\n")
        ok = [v for v in verdicts if v["consistent"] is not None]
        print(f"  judged {min(start + len(batch), len(todo))}/{len(todo)} ({time.perf_counter() - t0:.0f}s): "
              f"{sum(v['consistent'] for v in ok)}/{len(ok)} consistent"
              + (f", {len(batch) - len(ok)} not scored" if len(ok) < len(batch) else ""), flush=True)


# ---------------------------------------------------------------- summary
def _rate(num, den):
    return num / den if den else None


def summarise(name, rows, skipped=None):
    judged = [r for r in rows if r.get("consistent") is not None]
    out = {"name": name, "pairs": len(rows), "judged": len(judged),
           "consistency_rate": _rate(sum(r["consistent"] for r in judged), len(judged)),
           "same_answer_rate": _rate(sum(r["same_answer"] for r in judged), len(judged)),
           "contradiction_rate": _rate(sum(r["contradictions"] > 0 for r in judged), len(judged)),
           "differences_mean": _rate(sum(r["n_differences"] for r in judged), len(judged)),
           "judge_failed": sum(1 for r in rows if r.get("consistent") is None and r.get("judged_once")),
           "not_judged": sum(1 for r in rows if r.get("consistent") is None and not r.get("judged_once"))}
    if out["not_judged"]:                        # a rate over part of the pairs would read as the real one
        out.update(consistency_rate=None, same_answer_rate=None, contradiction_rate=None)
    if skipped is not None:
        out["not_compared"] = ", ".join(f"{k} {v}" for k, v in skipped.most_common()) or "-"
    return out


def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))

    def cell(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.0%}" if col.endswith("_rate") else f"{v:.2f}".rstrip("0").rstrip(".")
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
    parser = argparse.ArgumentParser(description="Answer consistency under rephrasing (LLM judge)")
    parser.add_argument("runs", nargs="*", help="robustness run ids (a name without _repN takes all its repeats)")
    parser.add_argument("--all", action="store_true", help="every robustness run")
    parser.add_argument("--baseline", required=True, help="normal run whose answers are the originals, e.g. a1_agentic_qwen")
    parser.add_argument("--data", default=str(DATASET), help="question set (default: src/evaluation/eval_questions.json)")
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE), help="judge config (default: agent_eval/configs/judge.yaml)")
    parser.add_argument("--no-judge", action="store_true", help="make no judge calls; summarise cached verdicts only")
    parser.add_argument("--workers", type=int, default=1, help="pairs judged at once (default 1 for a local judge)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed per judge call (default 900)")
    parser.add_argument("--attempts", type=int, default=3, help="runs that may retry a pair the judge failed on")
    parser.add_argument("--batch-size", type=int, default=10, help="pairs judged between saves (default 10)")
    args = parser.parse_args()
    if not args.runs and not args.all:
        parser.error("name runs to score, or use --all")

    dataset = {q["qid"]: q for q in json.loads(Path(args.data).read_text(encoding="utf-8"))}
    originals = baseline(args.baseline)
    if not originals:
        raise SystemExit(f"no answered records for {args.baseline!r} in {BASELINES}")
    runs = find_runs(args.runs, args.all)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read_judge(args.judge) if args.no_judge else load_judge(args.judge)
    key = f"{judge_name(cfg)}|{METHOD}"
    call = None if args.no_judge else json_judge(cfg, timeout=args.timeout)
    template = PROMPT.read_text(encoding="utf-8")

    table, groups = [], defaultdict(list)
    for run_id, records in runs.items():
        pairs, skipped = pairs_for(records, originals, dataset)
        if call is not None:
            judge_run(run_id, pairs, cfg, call, template, args.workers, args.batch_size, args.attempts)
        cached = {}
        for r in read_jsonl(OUT / f"{run_id}_consistency.jsonl"):
            if r.get("judge") == key and (r.get("consistent") is not None or r["case_id"] not in cached):
                cached[r["case_id"]] = r                 # a later success replaces an earlier failure
        rows = [{**p, **cached.get(p["case_id"], {}), "judged_once": p["case_id"] in cached} for p in pairs]
        table.append(summarise(run_id, rows, skipped))
        groups["all"] += rows
        for r in rows:
            groups[f"round {r['round']}"].append(r)
            groups[f"kind: {r['kind']}"].append(r)
        bad = [r for r in rows if r.get("consistent") is False]
        if bad:
            print(f"\n{run_id}: rephrasings whose answer differs from the original's")
            for r in sorted(bad, key=lambda r: -(r["contradictions"] or 0)):
                tag = "CONTRADICTS" if r["contradictions"] else "differs"
                first = (r["differences"] or [{}])[0]
                print(f"  {r['case_id']:<4} (of {r['variant_of']}, {r['kind']}) {tag}: {first.get('point', '')[:60]}"
                      f" | A: {first.get('answer_a', '')[:50]} | B: {first.get('answer_b', '')[:50]}")
    if len(runs) > 1:
        table.append(summarise("all runs pooled", groups["all"]))
    for k in sorted(g for g in groups if g.startswith("round ")):
        table.append(summarise(f"  {k}", groups[k]))
    for k in sorted(g for g in groups if g.startswith("kind: ")):
        table.append(summarise(f"  {k}", groups[k]))
    print()
    save_table("consistency", table,
               note=f"Judge `{judge_name(cfg)}` ({METHOD}) compares the answer to each rephrased question with the "
                    f"answer to the original (first answered repeat of `{args.baseline}`). Consistent = the judge says "
                    "a user gets the same answer from both and finds no contradiction. Only answerable pairs where "
                    "both versions were answered are compared; not_compared counts the rest.")


if __name__ == "__main__":
    main()