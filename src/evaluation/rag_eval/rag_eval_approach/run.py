"""Generate answers for one configuration and save them to results/outputs/<run_id>.jsonl (plan Section 9.3).

Run:  python src/evaluation/rag_eval/rag_eval_approach/run.py configs/a1_agentic_qwen.yaml --repeat 1
      python src/evaluation/rag_eval/rag_eval_approach/run.py configs/n1_naive.yaml --start 0 --end 12

- Every question is answered, including the out-of-scope ones.
- Resumable: questions already in the output file are skipped, so a crash or the cluster's 2-hour job limit only costs
  the question in progress. --start/--end take a slice of the dataset for one job.
- Never overwrites: a run id belongs to one configuration. If the config changes, the run refuses to continue;
  give the changed config a new run_id. --repeat N adds "_repN" (the agentic system is run 3 times).
- <run_id>.meta.json records the config and its hash, the git commit, the corpus snapshot and the machine.
- A question that raises is logged with status "error" and the run carries on; the scorers count it as wrong.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import yaml

from src import config
from src.evaluation.rag_eval.rag_eval_approach.answer import PROMPT_VERSION
from src.evaluation.rag_eval.rag_eval_approach.llm import resolve_model
from src.evaluation.rag_eval.rag_eval_approach.systems import build_system

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[1] / "eval_questions.json"
OUTPUTS = HERE.parent / "results" / "outputs"


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine():
    info = {"node": platform.node(), "platform": platform.platform()}
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda"] = None
    return info


def main():
    parser = argparse.ArgumentParser(description="Generate answers for one evaluation configuration")
    parser.add_argument("config", help="YAML config (a path, or a file name inside configs/)")
    parser.add_argument("--repeat", type=int, help="repeat number; the run id becomes <run_id>_rep<N>")
    parser.add_argument("--start", type=int, default=0, help="first question index (for splitting across jobs)")
    parser.add_argument("--end", type=int, help="question index to stop before")
    parser.add_argument("--data", default=str(DATASET))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = HERE / "configs" / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generator = cfg["generator"]   # resolve `default` before hashing, so the hash and the metadata name the real model
    generator["model"] = resolve_model(generator["provider"], generator["model"])
    config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    run_id = cfg["run_id"] + (f"_rep{args.repeat}" if args.repeat else "")
    today = dt.date.fromisoformat(str(cfg["today"]))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path, meta_path = OUTPUTS / f"{run_id}.jsonl", OUTPUTS / f"{run_id}.meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))["config_hash"]
        if previous != config_hash:
            sys.exit(f"{run_id} was started with a different config (hash {previous}, now {config_hash}). "
                     "Give the changed config a new run_id instead of mixing two configurations in one run.")

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    questions = data[args.start:args.end]
    done = set()
    if out_path.exists():
        done = {json.loads(line)["qid"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"{run_id}: {len(questions)} questions in slice, {len(questions) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    system = build_system(cfg, run_id)
    commit = git("rev-parse", "HEAD")
    if not meta_path.exists():
        collection = system.retriever.collection if hasattr(system, "retriever") else system.sup.retrieval.retriever.collection
        meta = {"run_id": run_id, "config": cfg, "config_file": config_path.name, "config_hash": config_hash,
                "local_models": {"intake": config.INTAKE_MODEL, "planner": config.PLANNER_MODEL,
                                 "supervisor": config.SUPERVISOR_MODEL, "auditor": config.AUDITOR_MODEL,
                                 "embedding": config.EMBED_MODEL, "reranker": config.RERANK_MODEL},
                "prompt_version": PROMPT_VERSION, "git_commit": commit, "git_dirty": bool(git("status", "--porcelain")),
                "corpus": {"collection": collection.name, "chunks": collection.count(),
                           "embed_model": (collection.metadata or {}).get("embed_model"),
                           "chroma_db_commit": git("log", "-1", "--format=%H", "--", "chroma_db"),
                           "chroma_db_dirty": bool(git("status", "--porcelain", "--", "chroma_db"))},
                "dataset": Path(args.data).name, "today": str(today), "machine": machine(),
                "started": dt.datetime.now().isoformat(timespec="seconds")}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    began = time.perf_counter()
    try:
        with out_path.open("a", encoding="utf-8") as out:
            for i, q in enumerate(todo, 1):
                start = time.perf_counter()
                try:
                    result, error = system.answer(q["question"], q["qid"], today), None
                except Exception as e:   # one bad question must not stop the run
                    result, error = {"status": "error", "answer": ""}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                record = {"run_id": run_id, "qid": q["qid"], "config_hash": config_hash, "git_commit": commit,
                          "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                          "generator": cfg["generator"]["model"], **result, "error": error,
                          "wall_s": time.perf_counter() - start}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                spent = time.perf_counter() - began
                eta = spent / i * (len(todo) - i)
                print(f"[{i}/{len(todo)}] {q['qid']} {record['status']:<10} {record['wall_s']:6.1f}s  "
                      f"elapsed {spent / 60:5.1f}m  ETA {eta / 60:5.1f}m  {q['question'][:50]}", flush=True)
    finally:
        system.close()


if __name__ == "__main__":
    main()
