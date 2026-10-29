"""Check that the configured judge works before any scoring: config, key, SDK, JSON output, and sanity of its grades.

Runs the real judge prompts (score_judge.py) on five hand-written answers whose right grades are obvious, built from the
dataset's own Q22 (penalty and GIRO instalment) and Q50 (out-of-scope GST):

    right answer              -> correctness 2, numeric_ok true
    wrong number (10% for 5%) -> correctness 0, numeric_ok false
    one part missing          -> correctness 1   (a warning only: partial credit is the judge's call)
    honest decline on Q50     -> refused true
    confident answer on Q50   -> refused false

This is a smoke test, not the validation in plan Section 4.3 (30 human-labelled pairs). It costs five judge calls.
Exit code 1 if the config is incomplete or a must-pass case fails.

Run:  python src/evaluation/rag_eval/rag_eval_approach/check_judge.py [--judge configs/judge.yaml]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import yaml

from src.evaluation.rag_eval.rag_eval_approach.env import has_key, load_env
from src.evaluation.rag_eval.rag_eval_approach.results_io import load_dataset
from src.evaluation.rag_eval.rag_eval_approach.score_judge import PROMPTS, ask, fill
from src.evaluation.rag_eval.rag_eval_approach.tracking import USAGE

HERE = Path(__file__).resolve().parent
SDK = {"ollama": "ollama", "anthropic": "anthropic", "openai": "openai", "groq": "openai", "google": "google.genai"}
KEY_NAME = {"ollama": "(none)", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY", "google": "GEMINI_API_KEY"}


def fail(message):
    print(f"FAIL  {message}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the configured judge")
    parser.add_argument("--judge", default=str(HERE / "configs" / "judge.yaml"))
    args = parser.parse_args()
    load_env()

    cfg = yaml.safe_load(Path(args.judge).read_text(encoding="utf-8"))
    provider, model = cfg["provider"], str(cfg["model"])
    if model.startswith("REPLACE"):
        fail(f"set `model` in {args.judge}")
    if provider not in SDK:
        fail(f"provider {provider!r} is not one of {sorted(SDK)}")
    print(f"ok    config: {provider}:{model}")
    if importlib.util.find_spec(SDK[provider].split(".")[0]) is None:
        fail(f"SDK for {provider} is not installed: pip install -r rag_eval_approach/requirements-api.txt")
    print("ok    SDK installed")
    if not has_key(provider):
        fail(f"no API key: put {KEY_NAME[provider]} in .env (copy .env.example) or export it")
    print("ok    API key found")

    # A judge that is also a compared generator rewards its own style (plan Section 4.1).
    for path in sorted((HERE / "configs").glob("*.yaml")):
        other = yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.name != "judge.yaml" and (other.get("generator") or {}).get("model") == model:
            print(f"WARN  the judge is also the generator in {path.name} (self-preference bias)")

    from src.evaluation.rag_eval.rag_eval_approach.llm import make_llm
    judge = make_llm(provider, model, **(cfg.get("options") or {}))
    templates = {kind: path.read_text(encoding="utf-8") for kind, path in PROMPTS.items()}
    data = load_dataset()
    q22, q50 = data["Q22"], data["Q50"]

    def grade(answer):
        prompt = fill(templates["correctness"], question=q22["question"], reference_answer=q22["reference_answer"],
                      key_facts="; ".join(q22["key_facts"]), response=answer)
        return ask(judge, "correctness", prompt)[0]

    def refusal(answer):
        return ask(judge, "refusal", fill(templates["refusal"], question=q50["question"], response=answer))[0]

    cases = [
        ("right answer", True, lambda: grade("A 5% late payment penalty is imposed on the unpaid tax. If you spread payment "
                                             "under the GIRO instalment plan, the minimum monthly deduction is $50."),
         lambda r: r["score"] == 2 and r["numeric_ok"] is True, "score 2, numeric_ok true"),
        ("wrong number", True, lambda: grade("A 10% late payment penalty is imposed on the unpaid tax. Under GIRO "
                                             "instalments the minimum monthly deduction is $50."),
         lambda r: r["score"] == 0 and r["numeric_ok"] is False, "score 0, numeric_ok false"),
        ("one part missing", False, lambda: grade("A 5% late payment penalty is imposed on the unpaid tax."),
         lambda r: r["score"] == 1, "score 1"),
        ("honest decline", True, lambda: refusal("The IRAS corporate tax material I have does not cover GST "
                                                 "registration, so I can't answer that."),
         lambda r: r["refused"] is True, "refused true"),
        ("confident answer", True, lambda: refusal("Yes. Once your taxable turnover exceeds $1 million you must "
                                                   "register for GST within 30 days."),
         lambda r: r["refused"] is False, "refused false"),
    ]
    USAGE.reset()
    failed = False
    for name, must_pass, run, passes, expect in cases:
        result = run()
        if result is None:
            status, got = "FAIL", "unparseable judge output"
        else:
            status = "ok  " if passes(result) else ("FAIL" if must_pass else "WARN")
            got = {k: v for k, v in result.items() if k != "reasoning"}
        failed |= status == "FAIL"
        print(f"{status}  {name:<17} expected {expect:<26} got {got}")
        if result and status != "ok  ":
            print(f"      judge said: {result['reasoning']}")
    use = USAGE.snapshot()
    print(f"\n{use['calls']} calls, {use['input']} input / {use['output']} output tokens")
    if failed:
        fail("the judge missed a must-pass case; try a stronger model or read the reasoning above before using it")
    print("\nOK  the judge connects and grades the obvious cases. Next: validate it on 30 human-labelled pairs (plan 4.3).")


if __name__ == "__main__":
    main()
