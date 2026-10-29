"""Paths and loaders shared by the scorers and analyze.py.

    results/outputs/<run_id>.jsonl        raw generations from run.py (never edited by hand)
    results/outputs/<run_id>.meta.json    config, commit, corpus snapshot and machine of the run
    results/scores/<run_id>_retrieval.jsonl, _judge.jsonl, _ragas.csv
"""
import json
from pathlib import Path

RAG_EVAL = Path(__file__).resolve().parents[1]
DATASET = RAG_EVAL.parent / "eval_questions.json"
RESULTS = RAG_EVAL / "results"
OUTPUTS = RESULTS / "outputs"
SCORES = RESULTS / "scores"

NON_ANSWERS = {"clarify", "refused", "no_answer", "error"}   # statuses that never count as a correct answer


def load_dataset(path=DATASET):
    return {q["qid"]: q for q in json.loads(Path(path).read_text(encoding="utf-8"))}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_ids(names=(), every=False):
    """The run ids to score: the ones named, or every run in results/outputs with --all."""
    if every:
        return sorted(p.stem for p in OUTPUTS.glob("*.jsonl"))
    missing = [n for n in names if not (OUTPUTS / f"{n}.jsonl").exists()]
    if missing:
        raise SystemExit(f"no outputs for {missing} in {OUTPUTS}")
    return list(names)


def load_meta(run_id):
    path = OUTPUTS / f"{run_id}.meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
