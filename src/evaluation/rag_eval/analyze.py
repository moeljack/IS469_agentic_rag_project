"""Turn the scores into the report's tables (plan Sections 7, 8 and 10). Reads only results/; calls no model.

    Table 1a     retrieval ablation R0-R4 (from rag_eval_retrieval/evaluate_rag.py)
    Runs         one row per run and per repeated config: accuracy with a bootstrap 95% CI, mean correctness, numeric
                 accuracy, refusals, non-answers, RAGAS means, retrieval, chunks seen/used, latency, tokens (Tables 1b, 2a, 3)
    --pair A B   paired comparison on the same questions (Table 2c): win/loss table, exact McNemar test, paired
                 bootstrap CI of the accuracy difference, latency and token ratios, and the plan's wording rule
    --types      accuracy by question type for the given runs (Table 2b)
    --stratify   accuracy split by whether that naive run's retrieval covered every fact (Table 2d)

A name is a run id (n1_naive) or a repeated config without its _repN suffix (a1_agentic_qwen = its repeats together:
per question, the share of repeats that were correct; the win/loss table uses the majority).

Run:  python src/evaluation/rag_eval/analyze.py
      python src/evaluation/rag_eval/analyze.py --pair a1_agentic_qwen n2_naive_matched --pair a1_agentic_qwen n1_naive
          --types n1_naive n2_naive_matched a1_agentic_qwen --stratify n1_naive
Tables are printed and saved as Markdown and CSV in results/report_assets/.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.evaluation.rag_eval.rag_eval_approach.results_io import (
    NON_ANSWERS, OUTPUTS, RESULTS, SCORES, load_dataset, read_jsonl)

ASSETS = RESULTS / "report_assets"
RETRIEVAL = RESULTS / "retrieval"
RAGAS_TEXT = ["user_input", "retrieved_contexts", "response", "reference", "judge"]


def bootstrap_ci(values, n_boot=1000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def load_run(run_id, dataset):
    """One row per question: the output record merged with its judge, retrieval and RAGAS scores."""
    rows = []
    for rec in read_jsonl(OUTPUTS / f"{run_id}.jsonl"):
        q = dataset[rec["qid"]]
        latency, tokens = rec.get("latency_s") or {}, rec.get("tokens") or {}
        rows.append({"run_id": run_id, "qid": rec["qid"], "answerable": bool(q.get("answerable")), "type": q["type"],
                     "qtype": q.get("qtype"), "source": q.get("source"), "numeric": bool(q.get("numeric")),
                     "status": "error" if rec.get("error") else rec.get("status"),
                     "latency": latency.get("total", rec.get("wall_s")),
                     "tokens": (tokens.get("input") or 0) + (tokens.get("output") or 0)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    judge = pd.DataFrame(read_jsonl(SCORES / f"{run_id}_judge.jsonl"))
    if not judge.empty:
        cols = [c for c in ("qid", "score", "numeric_ok", "refused") if c in judge]
        frame = frame.merge(judge[cols], on="qid", how="left")
    retrieval = pd.DataFrame(read_jsonl(SCORES / f"{run_id}_retrieval.jsonl"))
    if not retrieval.empty:
        frame = frame.merge(retrieval[["qid", "r", "hit", "n_seen", "n_used"]], on="qid", how="left")
    grounding = pd.DataFrame(read_jsonl(SCORES / f"{run_id}_grounding.jsonl"))
    if not grounding.empty:
        frame = frame.merge(grounding[["qid", "faithfulness", "ctx_precision", "ctx_ap"]], on="qid", how="left")
    ragas_path = SCORES / f"{run_id}_ragas.csv"
    if ragas_path.exists():
        ragas = pd.read_csv(ragas_path).drop(columns=RAGAS_TEXT, errors="ignore").drop_duplicates("qid", keep="last")
        frame = frame.merge(ragas.add_prefix("ragas_").rename(columns={"ragas_qid": "qid"}), on="qid", how="left")
    if "score" in frame:
        frame["correct"] = (frame["score"] == 2).astype(float).where(frame["answerable"])
    return frame


def summarise(name, frame):
    ans, oos = frame[frame["answerable"]], frame[~frame["answerable"]]
    row = {"name": name, "runs": frame["run_id"].nunique(), "answerable": ans["qid"].nunique(),
           "non_answers": int(ans["status"].isin(NON_ANSWERS).sum())}
    if "correct" in frame:
        per_q = ans.groupby("qid")["correct"].mean()
        lo, hi = bootstrap_ci(per_q.values)
        accs = ans.groupby("run_id")["correct"].mean()
        row.update(accuracy=per_q.mean(), ci_low=lo, ci_high=hi,
                   accuracy_range=f"{accs.min():.0%}-{accs.max():.0%}" if len(accs) > 1 else "",
                   mean_correctness=ans["score"].mean(),
                   numeric_accuracy=ans.loc[ans["numeric"], "numeric_ok"].astype(float).mean(),
                   unscored=int(ans["score"].isna().sum()))   # not judged yet, or judge output unparseable (counted wrong)
    if "refused" in frame and len(oos):
        row["refusals"] = f"{int((oos['refused'] == True).sum())}/{len(oos)}"   # unparseable (None) counts as not refused
    for col in [c for c in frame if c.startswith("ragas_") or c in ("faithfulness", "ctx_precision", "ctx_ap")]:
        row[col] = pd.to_numeric(ans[col], errors="coerce").mean()   # ragas_* only if the optional cross-check was run
    for col in ("r", "hit", "n_seen", "n_used"):
        if col in frame:
            row[{"r": "part_recall", "hit": "hit_all_parts", "n_seen": "chunks_seen", "n_used": "chunks_used"}[col]] = ans[col].mean()
    if "n_seen" in frame:
        row["chunks_seen_p90"] = ans["n_seen"].quantile(0.9)
    row.update(latency_median=frame["latency"].median(), latency_p90=frame["latency"].quantile(0.9),
               tokens_mean=frame["tokens"].mean())
    return row


def correctness(frames, name):
    """Per-question share of correct answers for a run id or a repeated config (answerable questions only)."""
    runs = [r for r in frames if r == name or base_name(r) == name]
    if not runs:
        raise SystemExit(f"no run called {name!r}; runs: {sorted(frames)}")
    frame = pd.concat([frames[r] for r in runs])
    if "correct" not in frame:
        raise SystemExit(f"{name}: no judge scores yet (run score_judge.py)")
    return frame[frame["answerable"]].groupby("qid")["correct"].mean(), frame


def wording(diff_questions, ci):
    if abs(diff_questions) <= 2:
        return "comparable; no claim of a winner"
    if ci[0] <= 0 <= ci[1]:
        return "slightly different, but within noise"
    return "higher" if diff_questions > 0 else "lower"


def pair(frames, a, b, seed=0):
    ca, fa = correctness(frames, a)
    cb, fb = correctness(frames, b)
    qids = sorted(set(ca.index) & set(cb.index))
    va, vb = ca[qids].values, cb[qids].values
    ma, mb = va > 0.5, vb > 0.5   # majority vote over repeats
    both, a_only, b_only, neither = int((ma & mb).sum()), int((ma & ~mb).sum()), int((~ma & mb).sum()), int((~ma & ~mb).sum())
    discordant = a_only + b_only
    p_value = binomtest(min(a_only, b_only), discordant, 0.5).pvalue if discordant else 1.0
    rng = np.random.default_rng(seed)
    diffs = [(va[idx] - vb[idx]).mean() for idx in (rng.integers(0, len(qids), len(qids)) for _ in range(1000))]
    ci = tuple(np.percentile(diffs, [2.5, 97.5]))
    diff = va.mean() - vb.mean()
    lat = fa["latency"].median() / fb["latency"].median() if fb["latency"].median() else float("nan")
    tok = fa["tokens"].mean() / fb["tokens"].mean() if fb["tokens"].mean() else float("nan")
    lines = [f"### {a} vs {b} ({len(qids)} answerable questions)", "",
             f"| | {b} correct | {b} not correct |", "|---|---|---|",
             f"| **{a} correct** | {both} | {a_only} |", f"| **{a} not correct** | {b_only} | {neither} |", "",
             f"- Accuracy: {va.mean():.0%} vs {vb.mean():.0%}; difference {diff:+.0%} "
             f"(paired bootstrap 95% CI {ci[0]:+.0%} to {ci[1]:+.0%}) = {wording(diff * len(qids), ci)}",
             f"- Exact McNemar test on the {discordant} discordant questions: p = {p_value:.3f}",
             f"- Latency ratio (median): {lat:.1f}x; token ratio (mean): {tok:.1f}x"]
    return "\n".join(lines)


def save(name, frame=None, text=None):
    ASSETS.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        frame.to_csv(ASSETS / f"{name}.csv", index=False)
        text = text or frame.to_string(index=False)
    (ASSETS / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n{text}\n  -> {ASSETS / (name + '.md')}")


def main():
    parser = argparse.ArgumentParser(description="Build the report tables from results/")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    parser.add_argument("--types", nargs="+", default=[], help="names for the accuracy-by-question-type table")
    parser.add_argument("--stratify", help="naive run whose retrieval hit splits the questions (Table 2d)")
    parser.add_argument("--runs", nargs="+", help="names for the stratified table (default: the --types names)")
    args = parser.parse_args()

    dataset = load_dataset()
    ablation = []
    for path in sorted(RETRIEVAL.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["summary"]["all"]
        ablation.append({"config": data["label"] + (" (paraphrases)" if data.get("paraphrases") else ""),
                         "n": s["n"], f"part_recall@{data['k']}": s["r"], f"precision@{data['k']}": s["p"],
                         f"F1@{data['k']}": s["f1"], "MRR": s["rr"], f"hit@{data['k']}": s["hit"],
                         "latency_ms_median": s["ms_median"]})
    if ablation:
        frame = pd.DataFrame(ablation)
        save("table1a_retrieval_ablation", frame)

    frames = {p.stem: load_run(p.stem, dataset) for p in sorted(OUTPUTS.glob("*.jsonl"))}
    frames = {k: v for k, v in frames.items() if not v.empty}
    if not frames:
        print("No generation outputs yet (results/outputs is empty).")
        return
    rows = [summarise(run_id, frame) for run_id, frame in frames.items()]
    groups = {}
    for run_id in frames:
        groups.setdefault(base_name(run_id), []).append(run_id)
    rows += [summarise(f"{name} (mean of {len(ids)})", pd.concat([frames[i] for i in ids]))
             for name, ids in groups.items() if len(ids) > 1]
    save("runs_summary", pd.DataFrame(rows))

    if args.pair:
        save("paired_comparisons", text="\n\n".join(pair(frames, a, b) for a, b in args.pair))

    if args.types:
        cells = {}
        for name in args.types:
            per_q, _ = correctness(frames, name)
            for qtype, group in pd.DataFrame({"correct": per_q}).join(
                    pd.Series({k: v.get("qtype") for k, v in dataset.items()}, name="qtype")).groupby("qtype"):
                cells.setdefault(qtype, {"type": qtype, "n": len(group)})[name] = (
                    f"{group['correct'].sum():.0f}/{len(group)} ({group['correct'].mean():.0%})")
        save("table2b_accuracy_by_type", pd.DataFrame(list(cells.values())))

    if args.stratify:
        hits = pd.DataFrame(read_jsonl(SCORES / f"{args.stratify}_retrieval.jsonl"))
        if hits.empty:
            raise SystemExit(f"no retrieval scores for {args.stratify} (run score_retrieval.py)")
        hit = hits.set_index("qid")["hit"]
        out = []
        for label, value in (("retrieval succeeded (hit = 1)", 1), ("retrieval missed (hit = 0)", 0)):
            qids = hit[hit == value].index
            row = {"stratum": label, "n": len(qids)}
            for name in args.runs or args.types or [args.stratify]:
                per_q, _ = correctness(frames, name)
                row[name] = per_q.reindex(qids).mean()
            out.append(row)
        save("table2d_stratified_by_naive_retrieval", pd.DataFrame(out))


if __name__ == "__main__":
    main()
