"""Correctness rubric and refusal checks with the LLM judge (plan Sections 3.3 and 4).

Answerable questions get a correctness score against the reference answer: 2 correct, 1 partial, 0 incorrect, plus
numeric_ok (every number in key_facts right). Failed runs and non-answers are scored 0 without calling the judge, and
stay in the denominator: a run error, an empty answer, or the status clarify / refused / no_answer.

Out-of-scope questions get a refusal check instead: refused true is correct. A no_answer or refused status, or a naive
answer starting with INSUFFICIENT_CONTEXT, counts as refused without the judge; clarify and run errors count as not
refused; any other answer goes to the judge with the short refusal prompt. `outcome` says what the system did:
declined | clarified | answered (a hallucination risk) | error | unscored, so a clarifying question is reported on its
own instead of being lumped in with answers.

Run:  python src/evaluation/rag_eval/rag_eval_approach/score_judge.py a1_agentic_qwen_rep1 n1_naive
      python src/evaluation/rag_eval/rag_eval_approach/score_judge.py --all --judge configs/judge.yaml
Writes results/scores/<run_id>_judge.jsonl. Already-scored questions are skipped (cached), so a judge fix means deleting
that file, never regenerating answers. The raw judge output is kept on every line.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import yaml

from src.evaluation.rag_eval.rag_eval_approach.answer import INSUFFICIENT
from src.evaluation.rag_eval.rag_eval_approach.llm import make_llm
from src.evaluation.rag_eval.rag_eval_approach.results_io import (
    DECLINED, NON_ANSWERS, OUTPUTS, SCORES, load_dataset, load_meta, oos_outcome, read_jsonl, run_ids)

HERE = Path(__file__).resolve().parent
PROMPTS = {"correctness": HERE / "prompts" / "judge_correctness_v1.txt", "refusal": HERE / "prompts" / "judge_refusal_v1.txt"}
SCHEMAS = {
    "correctness": {"type": "object", "properties": {"reasoning": {"type": "string"}, "numeric_ok": {"type": "boolean"},
                                                    "score": {"type": "integer", "enum": [0, 1, 2]}},
                    "required": ["reasoning", "numeric_ok", "score"]},
    "refusal": {"type": "object", "properties": {"reasoning": {"type": "string"}, "refused": {"type": "boolean"}},
                "required": ["reasoning", "refused"]},
}


def fill(template, **values):
    """str.format would trip over the JSON braces in the prompt, so placeholders are replaced one by one."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def parse(raw, kind):
    """The judge's JSON, or None. Tolerates code fences and text around the object."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and all(k in data for k in SCHEMAS[kind]["required"]):
            return data
    return None


def call(judge, schema, prompt, waits=(5, 15, 45, 90)):
    """One judge call, retried with a pause when the provider errors (503 overload, rate limit, dropped connection).
    After the last retry the error is raised, so the question is not saved as scored and the next run redoes it."""
    for wait in (*waits, None):
        try:
            return judge([{"role": "user", "content": prompt}], schema)
        except Exception as e:
            if wait is None or "PerDay" in str(e):   # a daily quota will not recover in seconds, so stop at once
                raise
            print(f"  judge call failed ({type(e).__name__}: {str(e)[:80]}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def ask(judge, kind, prompt):
    """(parsed, raw). Retries once on unparseable output, as the plan says."""
    raw = ""
    for _ in range(2):
        raw = call(judge, SCHEMAS[kind], prompt)
        parsed = parse(raw, kind)
        if parsed is not None:
            return parsed, raw
    return None, raw


def score(rec, q, judge, judge_name, templates):
    status, answer, error = rec.get("status"), (rec.get("answer") or "").strip(), rec.get("error")
    row = {"run_id": rec["run_id"], "qid": rec["qid"], "status": status, "judge": judge_name, "judged": False, "raw": None}
    if q.get("answerable"):
        row.update(kind="correctness", score=0, numeric_ok=False)
        if error:
            return {**row, "reason": "run error"}
        if not answer:
            return {**row, "reason": "empty answer"}
        if status in NON_ANSWERS or answer.startswith(INSUFFICIENT):
            return {**row, "reason": f"no answer (status {status})"}
        prompt = fill(templates["correctness"], question=q["question"], reference_answer=q["reference_answer"],
                      key_facts="; ".join(q.get("key_facts", [])) or "(none)", response=answer)
        parsed, raw = ask(judge, "correctness", prompt)
        if parsed is None:
            return {**row, "judged": True, "raw": raw, "reason": "judge output unparseable", "score": None}
        return {**row, "judged": True, "raw": raw, "reason": parsed["reasoning"],
                "score": int(parsed["score"]), "numeric_ok": bool(parsed["numeric_ok"])}

    row.update(kind="refusal", refused=False)
    if error:
        return {**row, "reason": "run error", "outcome": "error"}
    if status in DECLINED or answer.startswith(INSUFFICIENT):
        return {**row, "refused": True, "reason": f"declined (status {status})", "outcome": "declined"}
    if status == "clarify":
        return {**row, "reason": "asked a clarifying question instead of declining", "outcome": "clarified"}
    parsed, raw = ask(judge, "refusal", fill(templates["refusal"], question=q["question"], response=answer))
    refused = None if parsed is None else bool(parsed["refused"])
    return {**row, "judged": True, "raw": raw, "refused": refused, "outcome": oos_outcome(status, refused),
            "reason": "judge output unparseable" if parsed is None else parsed["reasoning"]}


def main():
    parser = argparse.ArgumentParser(description="Score generated answers with the LLM judge")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--all", action="store_true", help="every run in results/outputs")
    parser.add_argument("--judge", default=str(HERE / "configs" / "judge.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.judge).read_text(encoding="utf-8"))
    judge = make_llm(cfg["provider"], cfg["model"], **(cfg.get("options") or {}))
    judge_name = f"{cfg['provider']}:{cfg['model']}"
    templates = {kind: path.read_text(encoding="utf-8") for kind, path in PROMPTS.items()}
    dataset = load_dataset()
    SCORES.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids(args.runs, args.all):
        generator = load_meta(run_id).get("config", {}).get("generator", {})
        if generator.get("model") == cfg["model"]:
            print(f"WARNING {run_id}: the judge is the same model as the generator (self-preference bias)")
        out_path = SCORES / f"{run_id}_judge.jsonl"
        done = {r["qid"] for r in read_jsonl(out_path)}
        todo = [r for r in read_jsonl(OUTPUTS / f"{run_id}.jsonl") if r["qid"] not in done]
        print(f"{run_id}: {len(done)} already scored, {len(todo)} to score")
        with out_path.open("a", encoding="utf-8") as out:
            for rec in todo:
                row = score(rec, dataset[rec["qid"]], judge, judge_name, templates)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                result = row.get("score") if row["kind"] == "correctness" else row.get("refused")
                print(f"  {rec['qid']} {row['kind']:<11} {str(result):<6} {'judge' if row['judged'] else 'rule':<6} {row['reason'][:70]}")


if __name__ == "__main__":
    main()
