"""Faithfulness and context precision with the LLM judge: two calls per answer, in place of RAGAS (plan Section 3.2).

    faithfulness       share of the answer's factual claims that the retrieved documents support. The judge never sees
                       the reference answer, so it cannot excuse an unsupported claim because it matches the reference.
    context_precision  how much of the context was useful, from one call that marks every numbered chunk relevant or
                       not (using the reference answer). Reported two ways: `ctx_precision` is the share of chunks that
                       were relevant, `ctx_ap` is average precision (relevant chunks near the top score higher), the
                       quantity RAGAS's LLMContextPrecisionWithReference computes.

These are RAGAS-style scores, not RAGAS: call them that in the report, and cross-check about 10 answers with
score_ragas.py. Context recall is not computed; part recall against the gold passages already measures it with no judge.

Only answerable questions with a real answer are scored (a refusal or a failed run has nothing to ground). The contexts
are the chunks exactly as the answer prompt showed them (`context_used`).

Run:  python src/evaluation/rag_eval/rag_eval_approach/score_grounding.py a1_agentic_qwen_rep1 n1_naive
      python src/evaluation/rag_eval/rag_eval_approach/score_grounding.py --all
Writes results/scores/<run_id>_grounding.jsonl. Already-scored questions are skipped (cached); raw judge output is kept.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import yaml

from src.evaluation.rag_eval.rag_eval_approach.llm import make_llm
from src.evaluation.rag_eval.rag_eval_approach.results_io import (
    NON_ANSWERS, OUTPUTS, SCORES, load_dataset, load_meta, read_jsonl, run_ids)
from src.evaluation.rag_eval.rag_eval_approach.score_judge import call, fill

HERE = Path(__file__).resolve().parent
PROMPTS = {"faithfulness": HERE / "prompts" / "grounding_faithfulness_v1.txt",
           "precision": HERE / "prompts" / "grounding_context_precision_v1.txt"}
SCHEMAS = {
    "faithfulness": {"type": "object", "properties": {"claims": {"type": "array", "items": {
        "type": "object", "properties": {"claim": {"type": "string"}, "evidence": {"type": "string"},
                                         "supported": {"type": "boolean"}},
        "required": ["claim", "evidence", "supported"]}}}, "required": ["claims"]},
    "precision": {"type": "object", "properties": {"chunks": {"type": "array", "items": {
        "type": "object", "properties": {"index": {"type": "integer"}, "reason": {"type": "string"},
                                         "relevant": {"type": "boolean"}},
        "required": ["index", "reason", "relevant"]}}}, "required": ["chunks"]},
}
KEY = {"faithfulness": "claims", "precision": "chunks"}


def parse(raw, kind):
    """The judge's JSON, or None. Tolerates code fences and text around the object."""
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and isinstance(data.get(KEY[kind]), list):
            return data
    return None


def ask(judge, kind, prompt):
    """(parsed, raw). Retries once on unparseable output."""
    raw = ""
    for _ in range(2):
        raw = call(judge, SCHEMAS[kind], prompt)
        parsed = parse(raw, kind)
        if parsed is not None:
            return parsed, raw
    return None, raw


def context_text(chunks):
    return "\n\n".join(f"[Doc {i}]\n{c['text']}" for i, c in enumerate(chunks, 1))


def average_precision(flags):
    """Mean of precision@k at each relevant rank (0 if nothing is relevant)."""
    hits, total = 0, 0.0
    for k, flag in enumerate(flags, 1):
        if flag:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def score(rec, q, judge, judge_name, templates):
    chunks = rec["context_used"]
    context = context_text(chunks)
    row = {"run_id": rec["run_id"], "qid": rec["qid"], "judge": judge_name, "n_chunks": len(chunks)}

    parsed, raw = ask(judge, "faithfulness", fill(templates["faithfulness"], context=context, response=rec["answer"].strip()))
    row["faithfulness_raw"] = raw
    claims = (parsed or {}).get("claims")
    row["faithfulness"] = (sum(bool(c.get("supported")) for c in claims) / len(claims)) if claims else None
    row["n_claims"] = len(claims) if claims is not None else None
    row["unsupported"] = [c.get("claim") for c in claims or [] if not c.get("supported")]

    parsed, raw = ask(judge, "precision", fill(templates["precision"], question=q["question"],
                                               reference_answer=q["reference_answer"], context=context))
    row["precision_raw"] = raw
    marks = {int(c["index"]): bool(c["relevant"]) for c in (parsed or {}).get("chunks", []) if "index" in c}
    if parsed is not None and set(marks) >= set(range(1, len(chunks) + 1)):   # one verdict for every chunk
        flags = [marks[i] for i in range(1, len(chunks) + 1)]
        row.update(ctx_precision=sum(flags) / len(flags), ctx_ap=average_precision(flags), relevant=flags)
    else:
        row.update(ctx_precision=None, ctx_ap=None, relevant=None)
    return row


def main():
    parser = argparse.ArgumentParser(description="Faithfulness and context precision for generated runs")
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
        if load_meta(run_id).get("config", {}).get("generator", {}).get("model") == cfg["model"]:
            print(f"WARNING {run_id}: the judge is the same model as the generator (self-preference bias)")
        out_path = SCORES / f"{run_id}_grounding.jsonl"
        done = {r["qid"] for r in read_jsonl(out_path)}
        todo = []
        for rec in read_jsonl(OUTPUTS / f"{run_id}.jsonl"):
            q = dataset[rec["qid"]]
            answer = (rec.get("answer") or "").strip()
            if (rec["qid"] not in done and q.get("answerable") and not rec.get("error") and answer
                    and rec.get("status") not in NON_ANSWERS and rec.get("context_used")):
                todo.append(rec)
        print(f"{run_id}: {len(done)} already scored, {len(todo)} to score")
        with out_path.open("a", encoding="utf-8") as out:
            for i, rec in enumerate(todo, 1):
                row = score(rec, dataset[rec["qid"]], judge, judge_name, templates)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                f, p = row["faithfulness"], row["ctx_precision"]
                print(f"  [{i}/{len(todo)}] {rec['qid']}  faithfulness {'-' if f is None else format(f, '.2f'):>4} "
                      f"({row['n_claims']} claims)  context precision {'-' if p is None else format(p, '.2f'):>4}", flush=True)


if __name__ == "__main__":
    main()
