"""Cost and latency per question (Eval Plan: "Cost and Latency"; RQ4: what does each agentic RAG answer cost compared
with naive RAG?). No model is called: everything comes from the full runs in agent_eval/results/output/<run_id>.jsonl
(run_suite.py --suite main, for the agentic and the naive configs).

    RQ4a  token cost per question   `tokens` = input + output tokens summed over every LLM call for the question.
                                    For the agent that is the intake, route, decide, planner, auditor and answer
                                    calls; for naive RAG the one answer call. Both are counted by the same tracker
                                    (rag_eval_approach/tracking.py), so they compare. No price is applied: cost is
                                    reported in tokens (local models have no API price).
    RQ4b  wall-clock time per question   `latency_s.total`: from the question going in to the answer coming out
                                    (the run's own timer; `wall_s` is used if a record has no total).

Per run (results/cost_latency/cost_latency.{md,csv}):
    mean input / output / total tokens and LLM calls per question, with a bootstrap 95% CI on the mean total;
    median and p90 tokens; mean latency (RQ4b) with a 95% CI, plus median, p90 and max.
A config run several times (_rep1, _rep2, ...) also gets a pooled row: each question's value is first averaged over its
repeats, so every question counts once.

Agentic vs naive (results/cost_latency/cost_latency_pairs.md), on the questions both runs answered:
    token ratio and latency ratio (agentic mean / naive mean) with paired bootstrap 95% CIs (questions are resampled
    together), and the mean extra tokens and seconds per question with CIs.
Also: where the agent's time goes (mean seconds per stage: intake, route, retrieval, audit, decide, plan, answer), and
with --by-type, mean tokens and latency per question type and per status (answered vs refused / clarify / no_answer).

Questions whose run raised an error are left out (their tokens and time stop at the crash) and counted.

Run (anywhere, reads files only):
    python src/evaluation/agent_eval/score_cost_latency.py --all
    python src/evaluation/agent_eval/score_cost_latency.py a1_agentic_qwen n1_naive n2_naive_matched --by-type
    python src/evaluation/agent_eval/score_cost_latency.py --all --pair a1_agentic_qwen n1_naive --pair a1_agentic_qwen n2_naive_matched
Without --pair, every agentic config is paired with every naive config found.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from src.evaluation.agent_eval.paths import DATASET, OUTPUT as OUTPUTS, RESULTS  # noqa: E402

OUT = RESULTS / "cost_latency"
NOT_RUNS = ("_scores", "_judge", "_judge_details", "_trajectory", "_ragas", "_task_success", "_tool_calls",
            "_grounding", "_retrieval", "_consistency")


# ---------------------------------------------------------------- loading
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def find_runs(names=(), every=False, folder=OUTPUTS):
    files = sorted(p for p in Path(folder).glob("*.jsonl") if not p.stem.endswith(NOT_RUNS))
    if not every:
        files = [p for p in files if p.stem in names or base_name(p.stem) in names]
        found = {p.stem for p in files} | {base_name(p.stem) for p in files}
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"no outputs for {missing} in {folder}")
    runs = {p.stem: read_jsonl(p) for p in files}
    runs = {r: recs for r, recs in runs.items() if recs}
    if not runs:
        raise SystemExit(f"no runs in {folder}")
    return runs


def system_of(records):
    systems = Counter(r.get("system") for r in records if r.get("system"))
    return systems.most_common(1)[0][0] if systems else "?"


# ---------------------------------------------------------------- per question
def measure(rec):
    """{qid, status, input, output, total, calls, latency, stages} for one question, or None for a crashed one."""
    if rec.get("error") or rec.get("status") == "error":
        return None
    tokens = rec.get("tokens") or {}
    latency = rec.get("latency_s") or {}
    total_s = latency.get("total", rec.get("wall_s"))
    inp, out = tokens.get("input"), tokens.get("output")
    return {"qid": rec.get("qid"), "status": rec.get("status") or "",
            "input": inp, "output": out, "total": (inp or 0) + (out or 0) if inp is not None or out is not None else None,
            "calls": tokens.get("calls"), "latency": total_s,
            "stages": {k: v for k, v in latency.items() if k != "total" and isinstance(v, (int, float))}}


def per_question(records_by_run):
    """{qid: averaged measure} over several runs of one config (repeats): each question counts once."""
    groups = defaultdict(list)
    for records in records_by_run:
        for rec in records:
            m = measure(rec)
            if m is not None:
                groups[m["qid"]].append(m)
    out = {}
    for qid, ms in groups.items():
        row = {"qid": qid, "status": "/".join(sorted({m["status"] for m in ms})), "repeats": len(ms)}
        for k in ("input", "output", "total", "calls", "latency"):
            row[k] = _mean([m[k] for m in ms])
        stages = defaultdict(list)
        for m in ms:
            for s, v in m["stages"].items():
                stages[s].append(v)
        row["stages"] = {s: sum(v) / len(ms) for s, v in stages.items()}     # a stage missing in a repeat counts 0
        out[qid] = row
    return out


# ---------------------------------------------------------------- statistics
def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def bootstrap(values, stat, n_boot=2000, seed=0):
    """95% percentile interval of stat() over resampled lists (values: list of items, each resampled whole)."""
    import random
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        v = stat(sample)
        if v is not None and not math.isnan(v):
            draws.append(v)
    if not draws:
        return None, None
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def summarise(name, rows, system, runs=1, crashed=0):
    totals, lats = [r["total"] for r in rows], [r["latency"] for r in rows]
    tok_lo, tok_hi = bootstrap([t for t in totals if t is not None], _mean)
    lat_lo, lat_hi = bootstrap([v for v in lats if v is not None], _mean)
    return {"name": name, "system": system, "runs": runs, "questions": len(rows), "crashed_left_out": crashed,
            "tokens_in_mean": _mean([r["input"] for r in rows]), "tokens_out_mean": _mean([r["output"] for r in rows]),
            "tokens_mean": _mean(totals), "tokens_ci_low": tok_lo, "tokens_ci_high": tok_hi,
            "tokens_median": _pct(totals, 50), "tokens_p90": _pct(totals, 90),
            "llm_calls_mean": _mean([r["calls"] for r in rows]),
            "latency_mean_s": _mean(lats), "latency_ci_low_s": lat_lo, "latency_ci_high_s": lat_hi,
            "latency_median_s": _pct(lats, 50), "latency_p90_s": _pct(lats, 90),
            "latency_max_s": max((v for v in lats if v is not None), default=None),
            "no_token_counts": sum(r["total"] is None for r in rows)}


def pair(a_name, a_rows, b_name, b_rows):
    """Agentic (a) vs naive (b) on the questions both have: ratios of means and mean differences, paired bootstrap."""
    qids = sorted(set(a_rows) & set(b_rows))
    out = {"agentic": a_name, "naive": b_name, "questions": len(qids)}
    for metric, label in (("total", "tokens"), ("latency", "latency_s")):
        both = [(a_rows[q][metric], b_rows[q][metric]) for q in qids
                if a_rows[q][metric] is not None and b_rows[q][metric] is not None]
        if not both:
            continue

        def ratio(sample):
            den = sum(b for _, b in sample)
            return sum(a for a, _ in sample) / den if den else None

        def diff(sample):
            return sum(a - b for a, b in sample) / len(sample)
        r_lo, r_hi = bootstrap(both, ratio)
        d_lo, d_hi = bootstrap(both, diff)
        out.update({f"{label}_agentic_mean": _mean([a for a, _ in both]), f"{label}_naive_mean": _mean([b for _, b in both]),
                    f"{label}_ratio": ratio(both), f"{label}_ratio_ci": _ci(r_lo, r_hi, "x"),
                    f"{label}_extra_mean": diff(both), f"{label}_extra_ci": _ci(d_lo, d_hi, "")})
    return out


def _ci(lo, hi, unit):
    if lo is None:
        return ""
    fmt = (lambda v: f"{v:.2f}{unit}") if unit else (lambda v: f"{v:,.1f}")
    return f"{fmt(lo)} to {fmt(hi)}"


# ---------------------------------------------------------------- tables
def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))

    def cell(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            if col.endswith("_share"):
                return f"{v:.0%}"
            if col.endswith("_ratio"):
                return f"{v:.2f}x"
            if "tokens" in col and not col.endswith("_s"):
                return f"{v:,.0f}"
            return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:,.1f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(c, row.get(c)) for c in cols) + " |" for row in table]
    return "\n".join(lines)


def save(name, sections, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(f"### {title}\n\n{markdown(table)}" for title, table in sections if table)
    text += f"\n\n{note}" if note else ""
    (OUT / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    first = sections[0][1]
    cols = list(dict.fromkeys(k for row in first for k in row))
    lines = [",".join(cols)] + [",".join(json.dumps(row[c]) if isinstance(row.get(c), str)
                                         else ("" if row.get(c) is None else str(row[c])) for c in cols)
                                for row in first]
    (OUT / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(text)
    print(f"\n  -> {OUT / (name + '.md')}")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Tokens and wall-clock time per question, agentic vs naive (no LLM)")
    parser.add_argument("runs", nargs="*", help="run ids or configs (a name without _repN takes all its repeats)")
    parser.add_argument("--all", action="store_true", help="every run in agent_eval/results/output")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("AGENTIC", "NAIVE"),
                        help="compare two configs on the same questions (default: every agentic config vs every naive one)")
    parser.add_argument("--by-type", action="store_true", help="add tables per question type and per status")
    parser.add_argument("--data", default=str(DATASET), help="question set, for --by-type (default: eval_questions.json)")
    parser.add_argument("--outputs", default=str(OUTPUTS), help="folder with the run files (default: agent_eval/results/output)")
    args = parser.parse_args()
    if not args.runs and not args.all:
        parser.error("name runs to score, or use --all")

    runs = find_runs(args.runs, args.all, args.outputs)
    configs = defaultdict(list)
    for run_id in runs:
        configs[base_name(run_id)].append(run_id)
    systems = {c: system_of([r for m in members for r in runs[m]]) for c, members in configs.items()}

    table, per_config = [], {}
    for config, members in sorted(configs.items(), key=lambda kv: (systems[kv[0]] != "agentic", kv[0])):
        for run_id in members:
            rows = list(per_question([runs[run_id]]).values())
            crashed = sum(1 for r in runs[run_id] if measure(r) is None)
            table.append(summarise(run_id, rows, systems[config], crashed=crashed))
        pooled = per_question([runs[m] for m in members])
        per_config[config] = pooled
        if len(members) > 1:
            table.append(summarise(f"{config} (pooled {len(members)})", list(pooled.values()), systems[config],
                                   runs=len(members)))
    OUT.mkdir(parents=True, exist_ok=True)
    for config, rows in per_config.items():
        (OUT / f"{config}_cost_latency.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")

    # where the agent's time goes
    stage_rows = []
    for config, rows in per_config.items():
        if systems[config] != "agentic":
            continue
        lat = _mean([r["latency"] for r in rows.values()])
        names = sorted({s for r in rows.values() for s in r["stages"]})
        for s in names:
            v = _mean([r["stages"].get(s, 0.0) for r in rows.values()])
            stage_rows.append({"config": config, "stage": s, "mean_s": v, "time_share": v / lat if lat else None})

    # per question type / status
    type_rows, status_rows = [], []
    if args.by_type:
        dataset = {q["qid"]: q for q in json.loads(Path(args.data).read_text(encoding="utf-8"))}
        for config, rows in per_config.items():
            by_type, by_status = defaultdict(list), defaultdict(list)
            for r in rows.values():
                by_type[(dataset.get(r["qid"]) or {}).get("qtype") or "?"].append(r)
                by_status[r["status"]].append(r)
            for label, groups, sink in (("qtype", by_type, type_rows), ("status", by_status, status_rows)):
                for key in sorted(groups):
                    g = groups[key]
                    sink.append({"config": config, label: key, "questions": len(g),
                                 "tokens_mean": _mean([r["total"] for r in g]),
                                 "tokens_in_mean": _mean([r["input"] for r in g]),
                                 "tokens_out_mean": _mean([r["output"] for r in g]),
                                 "llm_calls_mean": _mean([r["calls"] for r in g]),
                                 "latency_mean_s": _mean([r["latency"] for r in g])})

    # agentic vs naive
    pairs = args.pair or [(a, b) for a in per_config if systems[a] == "agentic"
                          for b in per_config if systems[b] == "naive"]
    pair_rows = []
    for a, b in pairs:
        for name in (a, b):
            if name not in per_config:
                raise SystemExit(f"--pair: no run called {name!r} (configs: {sorted(per_config)})")
        pair_rows.append(pair(a, per_config[a], b, per_config[b]))

    sections = [("Per run (RQ4a: tokens per question; RQ4b: wall-clock seconds per question)", table)]
    if pair_rows:
        sections.append(("Agentic vs naive, same questions (ratio = agentic mean / naive mean; extra = agentic - naive "
                         "per question; paired bootstrap 95% CIs)", pair_rows))
    if stage_rows:
        sections.append(("Where the agent's time goes (mean seconds per question per stage)", stage_rows))
    if type_rows:
        sections.append(("By question type", type_rows))
    if status_rows:
        sections.append(("By status", status_rows))
    save("cost_latency", sections,
         note="Tokens = input + output tokens over every LLM call for the question (agent: intake, route, decide, "
              "planner, auditor, answer; naive: the answer call). No price is applied. Latency = the run's own "
              "timer from question to answer (latency_s.total, else wall_s). Pooled rows average each question over "
              "its repeats first. Questions whose run crashed are left out (crashed_left_out). Times depend on the "
              "hardware the runs used: compare runs made on the same machine.")


if __name__ == "__main__":
    main()