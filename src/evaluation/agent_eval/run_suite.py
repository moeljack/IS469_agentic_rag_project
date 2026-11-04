"""Run a system on the agent-eval test suites (Eval Plan: "Robustness and Safety") and save one record per case.

    robustness   data/robustness_cases.json: every rephrased question (P01-P05, the paraphrases that used to live in
                 the dataset, plus colloquial, typo and verbose rewrites of questions the agent answers).
    safety       data/injection_cases.json: prompt injections planted in a retrieved chunk (via injection.py) or
                 written into the user's message

Run (on the machine with Ollama and the retriever, like run.py):
    python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite robustness --repeat 1
    python src/evaluation/agent_eval/run_suite.py a1_agentic_qwen.yaml --suite safety --repeat 1
    python src/evaluation/agent_eval/run_suite.py n1_naive.yaml --suite safety        (naive baseline, for comparison)

The config is any run.py config (a path, or a file name in rag_eval_approach/configs/); the system is built exactly
as run.py builds it, only the retriever is wrapped for the safety suite. Output:
    rag_eval/results/agent_eval/<suite>/<run_id>_<suite>[_repN].jsonl   (+ .meta.json)
Resumable (cases already in the file are skipped) and never mixes configurations, like run.py. A case that raises
is saved with status "error". For the agent, the record also holds what the user store saved during the case
(`profile_facts`, `profile_history`), so the safety scorer can check whether an injection wrote to memory.
Score with score_robustness.py and score_safety.py (no model needed).
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.evaluation.rag_eval.rag_eval_approach.results_io import DATASET, RESULTS  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = ROOT / "src" / "evaluation" / "rag_eval" / "rag_eval_approach" / "configs"
CASE_FILES = {"robustness": HERE / "data" / "robustness_cases.json", "safety": HERE / "data" / "injection_cases.json"}
SUITE_OUT = RESULTS / "agent_eval"


def load_cases(suite, dataset_path=DATASET, case_file=None):
    """[{case_id, question, ...}] for the suite. Robustness cases get variant_of (the original qid) and kind."""
    cases = json.loads(Path(case_file or CASE_FILES[suite]).read_text(encoding="utf-8"))
    dataset = {q["qid"]: q for q in json.loads(Path(dataset_path).read_text(encoding="utf-8"))}
    if suite == "robustness":
        unknown = [c["case_id"] for c in cases if c["variant_of"] not in dataset]
    else:
        for c in cases:
            if c.get("vector") == "chunk" or not c.get("question"):
                c["question"] = dataset[c["base_qid"]]["question"]
        unknown = [c["case_id"] for c in cases if c.get("base_qid") and c["base_qid"] not in dataset]
    if unknown:
        raise SystemExit(f"cases refer to questions not in {Path(dataset_path).name}: {unknown}")
    return cases


def build(cfg, run_id, suite):
    """The system run.py would build, with the retriever wrapped for injection in the safety suite."""
    from src.evaluation.rag_eval.rag_eval_approach.systems import AgenticSystem, NaiveSystem
    from src.evaluation.rag_eval.rag_eval_retrieval.retrieval_modes import ModeRetriever
    retriever = ModeRetriever(cfg["retriever"]["mode"])
    if suite == "safety":
        from src.evaluation.agent_eval.injection import InjectingRetriever
        retriever = InjectingRetriever(retriever)
    if cfg["system"] == "agentic":
        return AgenticSystem(cfg, retriever, run_id), retriever
    if cfg["system"] == "naive":
        return NaiveSystem(cfg, retriever), retriever
    raise SystemExit(f"unknown system {cfg['system']!r}")


def memory_written(system, run_id, case_id):
    """What the agent's user store holds for this case's user after the turn (empty for the naive system)."""
    sup = getattr(system, "sup", None)
    if sup is None:
        return {}, []
    user_id = f"eval-{run_id}-{case_id}"   # AgenticSystem.answer() uses this id
    try:
        profile = sup.store.get_profile(user_id) or {}
        return dict(profile.get("facts") or {}), [{"field": h["field"], "value": h["value"]}
                                                  for h in sup.store.history(user_id)]
    except Exception as e:
        return {"_error": repr(e)}, []


def main():
    parser = argparse.ArgumentParser(description="Run the robustness or safety suite")
    parser.add_argument("config", help="run.py config: a path, or a file name in rag_eval_approach/configs/")
    parser.add_argument("--suite", required=True, choices=sorted(CASE_FILES))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--cases", help="a different case file for the suite")
    parser.add_argument("--only", nargs="+", help="run just these case ids")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = CONFIGS / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from src.evaluation.rag_eval.rag_eval_approach.llm import resolve_model
    cfg["generator"]["model"] = resolve_model(cfg["generator"]["provider"], cfg["generator"]["model"])
    config_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    run_id = f"{cfg['run_id']}_{args.suite}" + (f"_rep{args.repeat}" if args.repeat else "")
    today = dt.date.fromisoformat(str(cfg["today"]))

    cases = load_cases(args.suite, case_file=args.cases)
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only)]
    out_dir = SUITE_OUT / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, meta_path = out_dir / f"{run_id}.jsonl", out_dir / f"{run_id}.meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))["config_hash"]
        if previous != config_hash:
            sys.exit(f"{run_id} was started with a different config (hash {previous}, now {config_hash}); "
                     "use a new --repeat or run_id")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["case_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{run_id}: {len(cases)} cases, {len(cases) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return

    system, retriever = build(cfg, run_id, args.suite)
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"run_id": run_id, "suite": args.suite, "config": cfg,
                                         "config_file": config_path.name, "config_hash": config_hash,
                                         "case_file": str(args.cases or CASE_FILES[args.suite].name),
                                         "today": str(today),
                                         "started": dt.datetime.now().isoformat(timespec="seconds")}, indent=1),
                             encoding="utf-8")
    try:
        with out_path.open("a", encoding="utf-8") as out:
            for i, case in enumerate(todo, 1):
                if hasattr(retriever, "set_case"):
                    retriever.set_case(case)
                start = time.perf_counter()
                try:
                    result, error = system.answer(case["question"], case["case_id"], today), None
                except Exception as e:
                    result, error = {"status": "error", "answer": ""}, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                facts, history = memory_written(system, run_id, case["case_id"])
                record = {"run_id": run_id, "suite": args.suite, **case, "config_hash": config_hash,
                          "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                          "generator": cfg["generator"]["model"], **result,
                          "injected_ids": list(getattr(retriever, "poisoned_ids", [])),
                          "injections": getattr(retriever, "injections", 0),
                          "profile_facts": facts, "profile_history": history,
                          "error": error, "wall_s": time.perf_counter() - start}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(todo)}] {case['case_id']} {record['status']:<10} {record['wall_s']:6.1f}s  "
                      f"{case['question'][:60]}", flush=True)
    finally:
        if hasattr(retriever, "set_case"):
            retriever.set_case(None)
        system.close()


if __name__ == "__main__":
    main()
