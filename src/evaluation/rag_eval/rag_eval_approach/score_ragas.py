"""RAGAS metrics with the fixed judge model (plan Section 3.2). Runs in its own environment (requirements-scoring.txt):
ragas and its LangChain dependencies are kept out of the project environment, and this script only reads JSONL files.

    Faithfulness                        every claim in the answer is supported by the retrieved context
    ResponseRelevancy                   the answer addresses the question (needs embeddings: Qwen3-Embedding-0.6B)
    LLMContextPrecisionWithReference    retrieved chunks are relevant, and relevant ones rank high
    LLMContextRecall                    retrieved chunks cover everything in the reference answer

Only answerable questions with a real answer are scored (faithfulness is undefined for a refusal or a failed run).
The contexts are the chunks exactly as the answer prompt showed them (`context_used`), not everything retrieved.

Run:  python src/evaluation/rag_eval/rag_eval_approach/score_ragas.py a1_agentic_qwen_rep1 n1_naive
Writes results/scores/<run_id>_ragas.csv. Questions already in the file are skipped, so every score is cached.
RAGAS class names change between releases: these follow ragas 0.2+/0.3; pin the version you install.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.config import EMBED_MODEL_DIR
from src.evaluation.rag_eval.rag_eval_approach.results_io import NON_ANSWERS, OUTPUTS, SCORES, load_dataset, read_jsonl, run_ids

HERE = Path(__file__).resolve().parent


def chat_model(provider, model, options):
    """A LangChain chat model for RAGAS, using the same judge as score_judge.py."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=8000)   # no temperature: current Claude models reject it
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=options.get("temperature", 0))
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=options.get("temperature", 0))
    raise ValueError(f"no RAGAS judge for provider {provider!r}")


def main():
    parser = argparse.ArgumentParser(description="RAGAS metrics for generated runs")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--all", action="store_true", help="every run in results/outputs")
    parser.add_argument("--judge", default=str(HERE / "configs" / "judge.yaml"))
    parser.add_argument("--workers", type=int, default=4, help="parallel judge calls")
    args = parser.parse_args()

    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    from src.evaluation.rag_eval.rag_eval_approach.env import has_key, load_env
    load_env()
    cfg = yaml.safe_load(Path(args.judge).read_text(encoding="utf-8"))
    if not has_key(cfg["provider"]):
        raise SystemExit(f"no API key for {cfg['provider']!r}: add it to .env (see .env.example) or export it")
    if str(cfg["model"]).startswith("REPLACE"):
        raise SystemExit(f"set the judge model id in {args.judge}")
    llm = LangchainLLMWrapper(chat_model(cfg["provider"], cfg["model"], cfg.get("options") or {}))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=str(EMBED_MODEL_DIR), encode_kwargs={"normalize_embeddings": True}))
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    dataset = load_dataset()
    SCORES.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids(args.runs, args.all):
        out_path = SCORES / f"{run_id}_ragas.csv"
        done = set(pd.read_csv(out_path)["qid"]) if out_path.exists() else set()
        rows, qids = [], []
        for rec in read_jsonl(OUTPUTS / f"{run_id}.jsonl"):
            q = dataset[rec["qid"]]
            answer = (rec.get("answer") or "").strip()
            if (rec["qid"] in done or not q.get("answerable") or rec.get("error") or not answer
                    or rec.get("status") in NON_ANSWERS or not rec.get("context_used")):
                continue
            rows.append({"user_input": q["question"], "retrieved_contexts": [c["text"] for c in rec["context_used"]],
                         "response": answer, "reference": q["reference_answer"]})
            qids.append(rec["qid"])
        print(f"{run_id}: {len(done)} already scored, {len(rows)} to score")
        if not rows:
            continue
        result = evaluate(EvaluationDataset.from_list(rows), metrics=metrics, llm=llm, embeddings=embeddings,
                          run_config=RunConfig(max_workers=args.workers, timeout=300))
        frame = result.to_pandas()
        frame.insert(0, "qid", qids)
        frame.insert(1, "judge", f"{cfg['provider']}:{cfg['model']}")
        frame.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(frame.drop(columns=["user_input", "retrieved_contexts", "response", "reference"], errors="ignore")
              .mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
