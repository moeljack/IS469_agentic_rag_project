"""Run the whole RAG evaluation (plan Section 5) in order, one script after another.

    stage 1   retrieval ablation R0-R4 (evaluate_rag.py on the answerable questions), the plan's
              selection rule picks the retrieval mode for every later run, then answers with R0 and with that mode (N1)
    stage 2   the agent (A1) --repeats times, then N2 with k = A1's mean chunks in the answer prompt, rounded up
    stage 3   the API generators (configs/g_*.yaml), naive and agentic; a provider is skipped if its SDK is not
              installed or its model id is still REPLACE_...
    stage 4   score_retrieval, score_judge and score_grounding (faithfulness and context precision; both need a model in
              configs/judge.yaml), score_ragas only as an optional cross-check (--ragas-python, the scoring
              environment's python), then analyze.py with the plan's comparisons

Run:  python src/evaluation/rag_eval/run_all.py                      everything
      python src/evaluation/rag_eval/run_all.py --stages 1 2 --skip-api
      python src/evaluation/rag_eval/run_all.py --stages 4 --ragas-python scoringenv/Scripts/python.exe
      python src/evaluation/rag_eval/run_all.py --dry-run              print the commands only

Every step skips work that is already saved (run.py and the scorers resume), so after a crash or a cluster time limit
just run it again. Stage 2 needs stage 1's retrieval results; N2 waits until every A1 repeat has answered every question.
When the selected mode is not R4, the configs are copied with that mode into results/generated_configs/ under new
run ids (and the agent's reranker cutoff is switched off for modes without a reranker), so runs never mix configs.

Not automated, by design: setting the judge model, validating the judge against human labels (plan Section 4.3), and
verifying the needs_review reference answers. Do those before trusting stage 4.
"""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
APPROACH = HERE / "rag_eval_approach"
CONFIGS = APPROACH / "configs"
RESULTS = HERE / "results"
GENERATED = RESULTS / "generated_configs"
DATASET = HERE.parent / "eval_questions.json"
MODES = ["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"]
RERANKED = {"dense_rerank", "hybrid_rerank"}
SDK = {"anthropic": "anthropic", "openai": "openai", "groq": "openai", "google": "google.genai"}


STARTED = time.perf_counter()
STEPS = []   # (title, minutes) of the steps run so far, printed as a recap at the end


def step(title, command, dry_run, check=True):
    print(f"\n=== {title}   [{(time.perf_counter() - STARTED) / 60:.1f} min since start]\n"
          f"    {' '.join(str(c) for c in command)}", flush=True)
    if dry_run:
        return True
    began = time.perf_counter()
    result = subprocess.run([str(c) for c in command], cwd=ROOT)
    STEPS.append((title, (time.perf_counter() - began) / 60))
    print(f"--- done in {STEPS[-1][1]:.1f} min", flush=True)
    if result.returncode and check:
        sys.exit(f"stopped: '{title}' failed (exit {result.returncode}); fix it and run again, finished work is kept")
    return result.returncode == 0


def select_mode():
    """The plan's Stage 1 selection rule, written down before any run: carry R4 (hybrid + rerank) forward unless its
    part recall is below R0's, or its median latency is more than 3x R0's with no recall gain; otherwise carry the mode
    with the highest part recall, ties broken by MRR."""
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    expected = sum(1 for q in questions if q.get("answerable"))
    summary = {}
    for mode in MODES:
        path = RESULTS / "retrieval" / f"{mode}.json"
        if not path.exists():
            sys.exit(f"no stage 1 retrieval results for {mode} ({path}); run stage 1 first")
        summary[mode] = json.loads(path.read_text(encoding="utf-8"))["summary"]["all"]
        if summary[mode]["n"] != expected:   # e.g. left over from a --limit smoke test
            sys.exit(f"stage 1 results for {mode} cover {summary[mode]['n']} of {expected} questions; rerun stage 1")
    r4, r0 = summary["hybrid_rerank"], summary["dense"]
    if r4["r"] >= r0["r"] and not (r4["ms_median"] > 3 * r0["ms_median"] and r4["r"] <= r0["r"]):
        chosen, reason = "hybrid_rerank", "R4 keeps at least R0's part recall"
    else:
        chosen = max(MODES, key=lambda m: (summary[m]["r"], summary[m]["rr"]))
        reason = "R4 fails the rule; highest part recall (ties by MRR)"
    record = {"mode": chosen, "reason": reason, "part_recall": {m: summary[m]["r"] for m in MODES},
              "mrr": {m: summary[m]["rr"] for m in MODES}, "ms_median": {m: summary[m]["ms_median"] for m in MODES}}
    (RESULTS / "selected_mode.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"\nselected retrieval mode: {chosen} ({reason})")
    return chosen


def config_for(name, mode, **changes):
    """The config file to run: the original if nothing changes, else a copy with the mode and changes under a new run id."""
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    if mode == cfg["retriever"]["mode"] and not changes:
        return CONFIGS / f"{name}.yaml", cfg["run_id"]
    cfg["retriever"]["mode"] = mode
    if mode != "hybrid_rerank":
        cfg["run_id"] = f"{cfg['run_id']}_{mode}"
    if cfg["system"] == "agentic" and mode not in RERANKED:
        cfg["agent"]["audit_min_score"] = 0   # reranker cutoff means nothing without a reranker
    for key, value in changes.items():
        section, field = key.split(".")
        cfg[section][field] = value
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{cfg['run_id']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path, cfg["run_id"]


def answered(run_id):
    path = RESULTS / "outputs" / f"{run_id}.jsonl"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def main():
    parser = argparse.ArgumentParser(description="Run the whole RAG evaluation in order")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=1, help="A1 repeats (at temperature 0 the repeats were identical, so one is the default)")
    parser.add_argument("--skip-api", action="store_true", help="skip stage 3 even if the API SDKs are installed")
    parser.add_argument("--ragas-python", help="python of the scoring environment (requirements-scoring.txt) for RAGAS")
    parser.add_argument("--python", default=sys.executable, help="python of the project environment")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()
    py, dry = args.python, args.dry_run
    scored = len(json.loads(DATASET.read_text(encoding="utf-8")))

    if 1 in args.stages:
        evaluate = HERE / "rag_eval_retrieval" / "evaluate_rag.py"
        step("Stage 1: retrieval ablation R0-R4", [py, evaluate, "--mode", "all"], dry)
    mode = "hybrid_rerank" if dry and not (RESULTS / "retrieval" / "dense.json").exists() else select_mode()

    run = APPROACH / "run.py"
    n1_path, n1 = config_for("n1_naive", mode)
    s1_path, s1 = config_for("s1_naive_r0_dense", "dense")
    a1_path, a1 = config_for("a1_agentic_qwen", mode)
    if 1 in args.stages:
        step("Stage 1: answers with R0 dense", [py, run, s1_path], dry)
        if n1 != s1:
            step(f"Stage 1 / N1: answers with the selected mode ({mode})", [py, run, n1_path], dry)

    n2 = None
    if 2 in args.stages:
        for i in range(1, args.repeats + 1):
            step(f"Stage 2 / A1: agent, repeat {i} of {args.repeats}", [py, run, a1_path, "--repeat", i], dry)
        a1_runs = [f"{a1}_rep{i}" for i in range(1, args.repeats + 1)]
        if dry or all(answered(r) >= scored for r in a1_runs):
            step("Stage 2: retrieval scores of the A1 runs", [py, APPROACH / "score_retrieval.py", *a1_runs], dry)
            used = []
            for r in a1_runs:
                path = RESULTS / "scores" / f"{r}_retrieval.jsonl"
                used += [json.loads(line)["n_used"] for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
            k = max(1, math.ceil(sum(used) / len(used))) if used else 6
            print(f"\nN2 context size: k = {k} (A1's mean chunks in the answer prompt, rounded up)")
            n2_path, n2 = config_for("n2_naive_matched", mode, **{"retriever.k": k})
            step(f"Stage 2 / N2: naive with matched context (k = {k})", [py, run, n2_path], dry)
        else:
            print("\nN2 waits until every A1 repeat has answered every question; run again to continue.")

    stage3 = []
    if 3 in args.stages and not args.skip_api:
        for provider in ("anthropic", "openai"):
            names = [f"g_naive_{provider}", f"g_agentic_{provider}"]
            model = yaml.safe_load((CONFIGS / f"{names[0]}.yaml").read_text(encoding="utf-8"))["generator"]["model"]
            if str(model).startswith("REPLACE"):
                print(f"\nStage 3: skipping {provider}: set the model id in configs/{names[0]}.yaml and {names[1]}.yaml")
                continue
            if importlib.util.find_spec(SDK[provider]) is None:
                print(f"\nStage 3: skipping {provider}: its SDK is not installed (see requirements-api.txt)")
                continue
            for name in names:
                path, run_id = config_for(name, mode)
                step(f"Stage 3: {run_id}", [py, run, path], dry)
                stage3.append(run_id)

    if 4 in args.stages:
        step("Stage 4: retrieval scores", [py, APPROACH / "score_retrieval.py", "--all"], dry)
        judge = yaml.safe_load((CONFIGS / "judge.yaml").read_text(encoding="utf-8"))
        judged = not str(judge["model"]).startswith("REPLACE")
        if judged:
            step("Stage 4: correctness and refusal judge", [py, APPROACH / "score_judge.py", "--all"], dry)
            step("Stage 4: faithfulness and context precision", [py, APPROACH / "score_grounding.py", "--all"], dry)
        else:
            print("\nStage 4: skipping the judge: set provider and model in configs/judge.yaml")
        if args.ragas_python and judged:
            step("Stage 4: RAGAS cross-check", [args.ragas_python, APPROACH / "score_ragas.py", "--all"], dry, check=False)
        elif judged:
            print("\nStage 4: skipping RAGAS: pass --ragas-python <scoring environment's python>")

        command = [py, HERE / "analyze.py"]
        scores = RESULTS / "scores"
        has_judge = lambda run_id: dry or any(scores.glob(f"{run_id}*_judge.jsonl"))
        n2 = n2 or next((p.stem.replace("_judge", "") for p in scores.glob("n2_naive_matched*_judge.jsonl")), None)
        pairs = [(a1, n2), (a1, n1), (n1, s1)]
        pairs += [(f"g_agentic_{p}", f"g_naive_{p}") for p in ("anthropic", "openai")
                  if f"g_agentic_{p}" in stage3 or has_judge(f"g_agentic_{p}")]
        for a, b in pairs:
            if a and b and has_judge(a) and has_judge(b):
                command += ["--pair", a, b]
        main_runs = [r for r in (n1, n2, a1) if r and has_judge(r)]
        if main_runs:
            command += ["--types", *main_runs]
            if has_judge(n1) and (dry or (scores / f"{n1}_retrieval.jsonl").exists()):
                command += ["--stratify", n1]
        step("Stage 4: report tables", command, dry)


if __name__ == "__main__":
    main()
    if STEPS:
        print("\nSteps run this time:")
        for title, minutes in STEPS:
            print(f"  {minutes:6.1f} min  {title}")
        print(f"  {(time.perf_counter() - STARTED) / 60:6.1f} min  total")
