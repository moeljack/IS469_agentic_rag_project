"""Task success of the agent (Eval Plan: "Task Success"): did it complete the task correctly?

For an agentic RAG system a task is completed when the final answer is both faithful to the chunks the agent retrieved
(no hallucination) and relevant to the question asked. Both are scored by an LLM judge (LLM-as-a-judge, as the plan
asks), not by word overlap (BLEU/ROUGE). Two judge calls per answer, each returning JSON that the schema sent to
Ollama constrains, so a local 12B judge's replies always parse:

    faithfulness        share of the answer's factual claims supported by the retrieved chunks (RAGAS's definition).
                        The judge lists the claims and marks each supported or not (prompts/faithfulness_v1.txt,
                        copied from rag_eval's score_grounding.py). It never sees a reference answer.
    answer_relevancy    share of the parts of the question the answer addresses. The judge splits the question into
                        what it asks for and marks each part addressed or not (prompts/relevancy_v1.txt). Saying the
                        information is unavailable does not address a part.
RAGAS is not used: through LangChain it sends no JSON schema, and with the local gemma4:12b judge its faithfulness step
failed to parse on almost every answer. Report both as "RAGAS-style LLM-judge scores".

Per question (a "task"):
    answerable question     success = answered AND faithfulness >= --min-faithfulness AND answer_relevancy >=
                            --min-relevancy. A refusal, clarification, no_answer, error or empty answer is a failed
                            task (it stays in the denominator). An answer the judge could not score counts as failed
                            and is reported as judge_failed (retried on later runs). An answer without logged chunks
                            cannot be checked for faithfulness and counts as failed (answered_without_context).
    out-of-scope question   success = the agent declined (refused / no_answer / clarify); answering is a failure.
                            No judge call is made.
The chunks are the agent's `context_used`: exactly what the answer step was shown, logged by the harness
(rag_eval_approach/systems.py, AgenticSystem). The question given to the judge is the one the agent was asked.

Task success rate = share of answerable questions that succeeded, with a bootstrap 95% CI over questions. For a
config run several times (_rep1, _rep2, ...), each question's success is the mean over its repeats, as in analyze.py.
Out-of-scope decline rate and the overall rate over every question are reported beside it.

Thresholds: set them before looking at results and keep them fixed; both are written into every output.

Inputs (read only; nothing outside agent_eval is changed):
    --suite main         agent runs over the question set: rag_eval/results/outputs/<run_id>.jsonl (made by run.py)
    --suite robustness   rephrased questions: rag_eval/results/agent_eval/robustness/<run_id>.jsonl (run_suite.py);
                         each variant is judged against its own wording and scored as its original question
    --outputs DIR        read run files from another folder instead
Outputs (agent_eval/results/task_success/):
    <run_id>_judge.csv              judge scores, cached per item, judge and method: rerunning only scores what is new
    <run_id>_judge_details.jsonl    per answer: the judge's claims (which were unsupported) and question parts (which
                                    were not addressed), and the raw reply when a call failed
    <run_id>_task_success.jsonl     one row per question: status, scores, success, reason
    task_success_<suite>.md / .csv  the summary table

Run (needs only ollama, numpy, pandas and pyyaml, see requirements-scoring.txt):
    python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen
    python src/evaluation/agent_eval/score_task_success.py --all --by-type
    python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen_robustness --suite robustness
    python src/evaluation/agent_eval/score_task_success.py a1_agentic_qwen --no-judge     (summarise cached scores)
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
EVAL = HERE.parent                                          # src/evaluation
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.agent_eval.judge import (  # noqa: E402
    DEFAULT_JUDGE, json_judge, judge_name, load_judge, read_judge)

DATASET = EVAL / "eval_questions.json"
INPUTS = {"main": EVAL / "rag_eval" / "results" / "outputs",
          "robustness": EVAL / "rag_eval" / "results" / "agent_eval" / "robustness"}
OUT = HERE / "results" / "task_success"

DECLINED = {"refused", "no_answer", "clarify"}
SCORE_COLS = ("faithfulness", "answer_relevancy")

# The two judge prompts (faithfulness copied from rag_eval's grounding_faithfulness_v1.txt) and the JSON each returns
METHOD = "claims-v1+parts-v1"      # part of the cache key: scores from another method or prompt are never mixed in
PROMPTS = {"faithfulness": HERE / "prompts" / "faithfulness_v1.txt",
           "answer_relevancy": HERE / "prompts" / "relevancy_v1.txt"}
LIST_KEY = {"faithfulness": "claims", "answer_relevancy": "parts"}
VERDICT = {"faithfulness": "supported", "answer_relevancy": "addressed"}


def _schema(key, fields, verdict):
    props = {f: {"type": "string"} for f in fields}
    props[verdict] = {"type": "boolean"}
    return {"type": "object", "required": [key], "properties": {key: {"type": "array", "items": {
        "type": "object", "properties": props, "required": [*fields, verdict]}}}}


SCHEMAS = {"faithfulness": _schema("claims", ["claim", "evidence"], "supported"),
           "answer_relevancy": _schema("parts", ["part", "reason"], "addressed")}
NOT_RUNS = ("_scores", "_judge", "_judge_details", "_trajectory", "_ragas", "_task_success")


# ---------------------------------------------------------------- loading (copied from rag_eval results_io)
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path=DATASET):
    return {q["qid"]: q for q in json.loads(Path(path).read_text(encoding="utf-8"))}


def base_name(run_id):
    return re.sub(r"_rep\d+$", "", run_id)


def find_runs(folder, names=(), every=False, any_system=False):
    """{run_id: records}. A name without _repN takes all its repeats. --all keeps agentic runs only."""
    files = sorted(p for p in Path(folder).glob("*.jsonl") if not p.stem.endswith(NOT_RUNS))
    if not every:
        files = [p for p in files if p.stem in names or base_name(p.stem) in names]
        found = {p.stem for p in files} | {base_name(p.stem) for p in files}
        missing = [n for n in names if n not in found]
        if missing:
            raise SystemExit(f"no outputs for {missing} in {folder}")
    runs = {p.stem: read_jsonl(p) for p in files}
    if every and not any_system:
        skipped = sorted(r for r, recs in runs.items() if not any(x.get("system") == "agentic" for x in recs))
        if skipped:
            print(f"skipping non-agentic runs (use --any-system to include them): {skipped}")
        runs = {r: recs for r, recs in runs.items() if r not in skipped}
    if not runs:
        raise SystemExit(f"no runs to score in {folder}")
    return runs


def task(rec, dataset, suite):
    """(item_id, dataset question, the question as the agent was asked it)."""
    if suite == "robustness":
        q = dataset.get(rec.get("variant_of"))
        return rec.get("case_id"), q, rec.get("question") or (q or {}).get("question", "")
    q = dataset.get(rec.get("qid"))
    return rec.get("qid"), q, (q or {}).get("question", "")


def status_of(rec):
    return "error" if rec.get("error") else (rec.get("status") or "")


def judgeable(rec, q):
    """An answer the judge can score: answerable question, answered, non-empty answer, logged chunks."""
    return (bool(q and q.get("answerable")) and status_of(rec) == "answered"
            and bool((rec.get("answer") or "").strip()) and bool(rec.get("context_used")))


# ---------------------------------------------------------------- judging
def read_cache(path, judge):
    """{item_id: {faithfulness, answer_relevancy}} scored by this judge."""
    if not path.exists():
        return {}
    import pandas as pd
    frame = pd.read_csv(path, dtype={"item_id": str})
    frame = frame[frame["judge"] == judge].drop_duplicates("item_id", keep="last")
    return {row["item_id"]: {c: _num(row.get(c)) for c in SCORE_COLS} for _, row in frame.iterrows()}


def cache_key(cfg):
    """What the cached scores are keyed on: the judge model plus the scoring method (prompt versions)."""
    return f"{judge_name(cfg)}|{METHOD}"


def parse_items(raw, key):
    """The judge's list under `key` ("claims" / "parts"), or None. Tolerates code fences and text around the JSON."""
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return [c for c in data[key] if isinstance(c, dict)]
    return None


def ask(call, metric, prompt, tries=2):
    """(items, raw) from one judge call, retried once on an error or unparseable output. items None = failed."""
    raw = ""
    for _ in range(tries):
        try:
            raw = call(prompt, SCHEMAS[metric])
        except Exception as e:                                 # timeout, server error: try again, then give up
            raw = f"ERROR {type(e).__name__}: {e}"
            continue
        items = parse_items(raw, LIST_KEY[metric])
        if items is not None:
            return items, raw
    return None, raw


def faithfulness_of(call, template, contexts, answer):
    """(score, claims, raw): share of the answer's factual claims the chunks support. An answer with no factual claims
    scores 1.0 (nothing unsupported; answer relevance still judges whether it addressed the question)."""
    context = "\n\n".join(f"[Doc {i}]\n{text}" for i, text in enumerate(contexts, 1))
    claims, raw = ask(call, "faithfulness", template.replace("{context}", context).replace("{response}", answer))
    if claims is None:
        return None, None, raw
    return (sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else 1.0), claims, raw


def relevancy_of(call, template, question, answer):
    """(score, parts, raw): share of the question's parts the answer addresses. A reply with no parts is a failed
    judgement (every question asks for something), so it is retried like unparseable output."""
    parts, raw = ask(call, "answer_relevancy", template.replace("{question}", question).replace("{response}", answer))
    if not parts:
        return None, parts, raw
    return sum(bool(p.get("addressed")) for p in parts) / len(parts), parts, raw


def make_scorer(cfg, workers=1, timeout=900):
    """score(rows) -> (frame with faithfulness + answer_relevancy, per-row details). Two judge calls per answer.
    workers: answers judged at the same time. Keep 1 with one local Ollama server: it answers one request at a time,
    so parallel requests only wait in its queue; raise it for an API judge."""
    import pandas as pd
    call = json_judge(cfg, timeout=timeout)
    templates = {m: p.read_text(encoding="utf-8") for m, p in PROMPTS.items()}

    def one(row):
        f, claims, f_raw = faithfulness_of(call, templates["faithfulness"], row["contexts"], row["answer"])
        r, parts, r_raw = relevancy_of(call, templates["answer_relevancy"], row["question"], row["answer"])
        return {"faithfulness": f, "answer_relevancy": r,
                "n_claims": None if claims is None else len(claims),
                "unsupported": [c.get("claim") for c in claims or [] if not c.get("supported")],
                "not_addressed": [p.get("part") for p in parts or [] if not p.get("addressed")],
                "claims": claims, "parts": parts,
                "faithfulness_raw": None if f is not None else f_raw[:2000],
                "relevancy_raw": None if r is not None else r_raw[:2000]}

    def score(rows):
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                details = list(pool.map(one, rows))
        else:
            details = [one(row) for row in rows]
        return pd.DataFrame({c: [d[c] for d in details] for c in SCORE_COLS}), details
    return score


def read_attempts(path, judge):
    """{item_id: how many times this judge has been asked to score it} (one cached row per attempt)."""
    if not path.exists():
        return {}
    import pandas as pd
    frame = pd.read_csv(path, dtype={"item_id": str})
    return frame[frame["judge"] == judge]["item_id"].value_counts().to_dict()


def judge_run(run_id, records, dataset, suite, cfg, scorer, batch_size=10, attempts=3):
    """Score the run's judgeable answers that are not cached yet; append them to <run_id>_judge.csv after every
    batch, so a job stopped by the cluster's time limit keeps what it judged and a rerun carries on from there.
    An answer the judge returned no score for (timeout, unparseable output) is judged again on the next run, up to
    `attempts` times in all; after that it stays judge_failed and counts as a failed task."""
    path = OUT / f"{run_id}_judge.csv"
    details_path = OUT / f"{run_id}_judge_details.jsonl"
    name = cache_key(cfg)
    cache, tries = read_cache(path, name), read_attempts(path, name)
    scored = {i for i, s in cache.items() if all(s.get(c) is not None for c in SCORE_COLS)}
    gave_up = {i for i in cache if i not in scored and tries.get(i, 0) >= attempts}
    done = scored | gave_up
    retry = sorted(set(cache) - done)
    if retry:
        print(f"{run_id}: judging again {len(retry)} answer(s) the judge returned no score for: {retry[:10]}")
    rows, ids = [], []
    for rec in records:
        item_id, q, question = task(rec, dataset, suite)
        if item_id in done or item_id in ids or not judgeable(rec, q):
            continue
        rows.append({"question": question, "answer": rec["answer"].strip(),
                     "contexts": [c["text"] for c in rec["context_used"]]})
        ids.append(item_id)
    print(f"{run_id}: {len(scored)} answers already judged by {name}, {len(gave_up)} failed {attempts} times "
          f"(left as judge_failed), {len(rows)} to judge")
    step = max(1, batch_size)
    import pandas as pd
    for start in range(0, len(rows), step):
        t0 = time.perf_counter()
        batch_ids = ids[start:start + step]
        frame, details = scorer(rows[start:start + step])
        frame.insert(0, "item_id", batch_ids)
        frame.insert(1, "judge", name)
        frame.reindex(columns=["item_id", "judge", *SCORE_COLS]).to_csv(
            path, mode="a", header=not path.exists(), index=False)
        with details_path.open("a", encoding="utf-8") as out:   # claims and parts, to read why an answer failed
            for item_id, d in zip(batch_ids, details):
                out.write(json.dumps({"item_id": item_id, "judge": name, **d}, ensure_ascii=False) + "\n")
        means = frame[list(SCORE_COLS)].apply(pd.to_numeric, errors="coerce").mean()
        failed = int(frame[list(SCORE_COLS)].isna().any(axis=1).sum())
        print(f"  judged {min(start + step, len(rows))}/{len(rows)} ({time.perf_counter() - t0:.0f}s): "
              + ", ".join(f"{c} {v:.3f}" for c, v in means.items())
              + (f", {failed} not scored" if failed else ""), flush=True)


# ---------------------------------------------------------------- task outcomes
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def outcome(rec, dataset, suite, scores, min_faith, min_rel):
    """One row per task: {item_id, qid, answerable, qtype, status, faithfulness, answer_relevancy, success, reason}.
    success is None only when the answer still has to be judged (run without --no-judge first)."""
    item_id, q, _ = task(rec, dataset, suite)
    status = status_of(rec)
    row = {"item_id": item_id, "qid": (q or {}).get("qid"), "answerable": bool(q and q.get("answerable")),
           "qtype": (q or {}).get("qtype"), "type": (q or {}).get("type"), "kind": rec.get("kind"),
           "status": status, "faithfulness": None, "answer_relevancy": None, "success": False, "reason": ""}
    if not row["answerable"]:
        row["success"] = status in DECLINED
        row["reason"] = "declined" if row["success"] else ("answered_out_of_scope" if status == "answered" else status)
        return row
    if status != "answered":
        row["reason"] = f"not_answered:{status or 'unknown'}"
    elif not (rec.get("answer") or "").strip():
        row["reason"] = "empty_answer"
    elif not rec.get("context_used"):
        row["reason"] = "answered_without_context"
    elif item_id not in scores:
        row.update(success=None, reason="not_judged")
    else:
        faith, rel = scores[item_id].get("faithfulness"), scores[item_id].get("answer_relevancy")
        row.update(faithfulness=faith, answer_relevancy=rel)
        if faith is None or rel is None:
            row["reason"] = "judge_failed"
        else:
            faithful, relevant = faith >= min_faith, rel >= min_rel
            row["success"] = faithful and relevant
            row["reason"] = ("success" if row["success"] else
                             "+".join(r for r, bad in (("unfaithful", not faithful), ("irrelevant", not relevant)) if bad))
    return row


# ---------------------------------------------------------------- summary
def bootstrap_ci(values, n_boot=1000, seed=0):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _per_item(rows):
    """{item_id: mean success over its repeats}, leaving out tasks not judged yet."""
    groups = defaultdict(list)
    for r in rows:
        if r["success"] is not None:
            groups[r["item_id"]].append(float(r["success"]))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def summarise(name, rows, runs=1):
    ans = [r for r in rows if r["answerable"]]
    oos = [r for r in rows if not r["answerable"]]
    per_item = _per_item(ans)
    lo, hi = bootstrap_ci(list(per_item.values()))
    judged = [r for r in ans if r["faithfulness"] is not None and r["answer_relevancy"] is not None]
    out = {"name": name, "runs": runs, "answerable": len({r["item_id"] for r in ans}),
           "task_success_rate": _mean(list(per_item.values())), "ci_low": lo, "ci_high": hi,
           "answered_rate": _mean([float(r["status"] == "answered") for r in ans]),
           "judged": len(judged),
           "faithfulness_mean": _mean([r["faithfulness"] for r in judged]),
           "answer_relevancy_mean": _mean([r["answer_relevancy"] for r in judged]),
           "faithful_rate": _mean([float("unfaithful" not in r["reason"]) for r in judged]),
           "relevant_rate": _mean([float("irrelevant" not in r["reason"]) for r in judged]),
           "judge_failed": sum(r["reason"] == "judge_failed" for r in ans),
           "not_judged": sum(r["reason"] == "not_judged" for r in ans)}
    if out["not_judged"]:                       # a rate over part of the questions would read as the real one
        out.update(task_success_rate=None, ci_low=None, ci_high=None)
    if runs > 1:
        by_run = defaultdict(list)
        for r in ans:
            if r["success"] is not None:
                by_run[r["run_id"]].append(float(r["success"]))
        rates = [sum(v) / len(v) for v in by_run.values() if v]
        out["success_range"] = f"{min(rates):.0%}-{max(rates):.0%}" if rates else ""
    if oos:
        out["out_of_scope"] = len({r["item_id"] for r in oos})
        out["oos_decline_rate"] = _mean([float(r["success"]) for r in oos])
        everything = _per_item(rows)
        out["overall_success_rate"] = None if out["not_judged"] else _mean(list(everything.values()))
    fails = Counter(r["reason"] for r in rows if r["success"] is False)
    out["failures"] = ", ".join(f"{k} {v}" for k, v in fails.most_common()) or "-"
    return out


def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))

    def cell(col, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            if col.endswith("_rate") or col in ("ci_low", "ci_high"):
                return f"{v:.0%}"
            return f"{v:.3f}".rstrip("0").rstrip(".")
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


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Task success of the agent: LLM-judged faithfulness + answer relevance")
    parser.add_argument("runs", nargs="*", help="run ids (a name without _repN takes all its repeats)")
    parser.add_argument("--all", action="store_true", help="every agentic run in the input folder")
    parser.add_argument("--suite", choices=sorted(INPUTS), default="main")
    parser.add_argument("--outputs", help="folder with the run .jsonl files (default: depends on --suite)")
    parser.add_argument("--data", default=str(DATASET), help="question set (default: src/evaluation/eval_questions.json)")
    parser.add_argument("--judge", default=str(DEFAULT_JUDGE), help="judge config (default: agent_eval/configs/judge.yaml)")
    parser.add_argument("--min-faithfulness", type=float, default=0.8)
    parser.add_argument("--min-relevancy", type=float, default=0.7)
    parser.add_argument("--no-judge", action="store_true", help="make no judge calls; summarise cached scores only")
    parser.add_argument("--workers", type=int, default=1,
                        help="answers judged at the same time (default 1: one local Ollama server answers one request "
                             "at a time, so more only wait in its queue; raise it for an API judge)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed per judge call (default 900)")
    parser.add_argument("--attempts", type=int, default=3,
                        help="runs that may try an answer the judge returned no score for (default 3)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="answers judged per batch; each batch is saved before the next starts (default 10)")
    parser.add_argument("--any-system", action="store_true", help="with --all, also score naive runs (comparison)")
    parser.add_argument("--by-type", action="store_true", help="add rows per question type (qtype)")
    args = parser.parse_args()
    if not args.runs and not args.all:
        parser.error("name runs to score, or use --all")

    dataset = load_dataset(args.data)
    runs = find_runs(args.outputs or INPUTS[args.suite], args.runs, args.all, args.any_system)
    for run_id, records in runs.items():
        unknown = sorted({str(task(r, dataset, args.suite)[0]) for r in records if task(r, dataset, args.suite)[1] is None})
        if unknown:
            print(f"warning {run_id}: {len(unknown)} record(s) refer to questions not in {Path(args.data).name}, "
                  f"left out: {unknown[:10]}")
            runs[run_id] = [r for r in records if task(r, dataset, args.suite)[1] is not None]

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read_judge(args.judge) if args.no_judge else load_judge(args.judge)
    if not args.no_judge:
        scorer = make_scorer(cfg, args.workers, args.timeout)
        for run_id, records in runs.items():
            judge_run(run_id, records, dataset, args.suite, cfg, scorer, args.batch_size, args.attempts)

    name = cache_key(cfg)
    table, groups = [], defaultdict(list)
    for run_id, records in runs.items():
        scores = read_cache(OUT / f"{run_id}_judge.csv", name)
        rows = [{"run_id": run_id, **outcome(rec, dataset, args.suite, scores, args.min_faithfulness,
                                             args.min_relevancy)} for rec in records]
        (OUT / f"{run_id}_task_success.jsonl").write_text(
            "".join(json.dumps({**r, "judge": name, "min_faithfulness": args.min_faithfulness,
                                "min_relevancy": args.min_relevancy}) + "\n" for r in rows), encoding="utf-8")
        table.append(summarise(run_id, rows))
        groups[base_name(run_id)].append(rows)
        waiting = sum(r["reason"] == "not_judged" for r in rows)
        if waiting:
            print(f"note {run_id}: {waiting} answer(s) not judged yet and left out of the rate; run without --no-judge")
        failed = sum(r["reason"] == "judge_failed" for r in rows)
        if failed:
            print(f"note {run_id}: the judge returned no score for {failed} answer(s); they count as failed tasks")

    pooled = {}
    for base, members in groups.items():
        rows = [r for m in members for r in m]
        pooled[base] = rows
        if len(members) > 1:
            table.append(summarise(f"{base} (pooled {len(members)})", rows, runs=len(members)))
    if args.by_type:
        for base, rows in pooled.items():
            for qtype in sorted({str(r["qtype"]) for r in rows}):
                part = [r for r in rows if str(r["qtype"]) == qtype]
                table.append({"name": f"  {base} qtype: {qtype}", "runs": len({r["run_id"] for r in part}),
                              **{k: v for k, v in summarise("", part).items()
                                 if k in ("answerable", "task_success_rate", "out_of_scope", "oos_decline_rate",
                                          "overall_success_rate", "failures")}})
    print()
    save_table(f"task_success_{args.suite}", table,
               note=f"Judge `{judge_name(cfg)}`, RAGAS-style LLM-judge scores ({METHOD}). Faithfulness: share of the "
                    "answer's claims the retrieved chunks support. Answer relevancy: share of the question's parts the "
                    "answer addresses. Success on an answerable question = "
                    f"answered, faithfulness >= {args.min_faithfulness} and answer relevancy >= {args.min_relevancy}; "
                    "refusals, clarifications, no_answer, errors and unscorable answers count as failures. "
                    "Out-of-scope success = declined. CI: bootstrap 95% over questions.")


if __name__ == "__main__":
    main()