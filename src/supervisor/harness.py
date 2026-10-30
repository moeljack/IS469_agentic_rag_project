"""Wire the real components together and chat in the terminal.

    python src/supervisor/harness.py --user acme            # interactive
    python src/supervisor/harness.py --user acme --trace    # also print what each agent did
    python src/supervisor/harness.py --user acme --forget   # delete everything stored for this user

Company details can be set once:  --company-name "Acme Pte Ltd" --uen 201912345A

Speed (all optional):
  --k, --max-chunks, --max-queries   shrink the retrieval work per question
  --candidates N                     chunks the reranker scores per query (default 6; 0 = CANDIDATES in src/config.py)
  --audit-min-score S                chunks the reranker scores below S skip the auditor (default 0.5; 0 = audit all)
  --quantize-rerank                  int8 reranker weights: faster on CPU only (leave off when torch uses Apple MPS)
  --no-warm                          don't preload the Ollama models and the reranker at start-up
--trace ends every answer with a per-stage timing line.

Decomposition:
  default                            intake gate only extracts + refuses/clarifies; the supervisor's route step
                                     decides whether to search the question as written or decompose it first
  --intake-plans                     the original pipeline: the intake call also writes the sub-queries and step 1
                                     searches them (no route step). Use it to compare the two on the eval set.

Pipeline:  intake gate (qwen3.5:4b)  ->  Supervisor (qwen3.5:4b)
                                              |-- route: search as written | decompose (query planner, qwen3.5:4b)
                                              |-- RetrievalAgent -> retriever.Retriever (ChromaDB + BM25 + reranker)
                                              |-- Auditor        -> auditor.AuditorAgent (qwen3.5:4b, ollama client)
                                              '-- UserStore (ChromaDB profile + pending clarifications)
Needs: ollama running with qwen3.5:4b pulled (all models use the `ollama` Python client), and the ChromaDB
collection built by the ingest scripts.
Models: SUPERVISOR_MODEL / INTAKE_MODEL / PLANNER_MODEL / AUDITOR_MODEL all default to qwen3.5:4b, so one model
stays loaded.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supervisor import (
    AUDIT_MIN_SCORE, AUDITOR_MODEL, DEFAULT_CANDIDATES, DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES, MODEL,
    Auditor, RetrievalAgent, Supervisor, pin_ollama_keep_alive, warm_models,
)
from src.retrieval.user_store import UserStore


def build_supervisor(max_steps: int = 4, tools: dict = None, store: UserStore = None, k: int = DEFAULT_K,
                     max_chunks: int = DEFAULT_MAX_CHUNKS, max_queries: int = DEFAULT_MAX_QUERIES,
                     candidates: int = DEFAULT_CANDIDATES, audit_min_score: float = AUDIT_MIN_SCORE,
                     retriever=None, intake_plans: bool = False) -> Supervisor:
    """The real stack. Retriever and AuditorAgent are wrapped by the adapters in supervisor.py, which give them the
    .run() / .audit() interface the supervisor loop uses. The decomposer's intake gate is the Supervisor's default
    intake and its query planner the default planner.
    candidates <= 0 (or None) leaves the pool size to src/config.py's CANDIDATES."""
    if retriever is None:
        # imported here, not at the top: src.config reads RERANK_QUANTIZE when the retriever module is first
        # imported, and main() sets it from --quantize-rerank before calling this
        from src.retrieval.retriever import Retriever
        retriever = Retriever()
    return Supervisor(store=store or UserStore(),
                      retrieval=RetrievalAgent(retriever, k=k, max_chunks=max_chunks,
                                               candidates=candidates if candidates and candidates > 0 else None),
                      auditor=Auditor(),                         # AuditorAgent is created on first audit
                      tools=tools, max_steps=max_steps, max_queries=max_queries, audit_min_score=audit_min_score,
                      intake_plans=intake_plans)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="user id (one profile per id)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4, help="max supervisor actions before it must answer")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="chunks kept per query after reranking")
    ap.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS, help="chunks the auditor judges per search")
    ap.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES, help="queries run per search")
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                    help="candidates the reranker scores per query; lower = faster (0 = CANDIDATES in src/config.py)")
    ap.add_argument("--audit-min-score", type=float, default=AUDIT_MIN_SCORE,
                    help="reranker score below which a chunk is rejected without an auditor call (0 = audit all)")
    ap.add_argument("--quantize-rerank", action="store_true",
                    help="int8 reranker weights (sets RERANK_QUANTIZE=1): faster on CPU, not for Apple MPS")
    ap.add_argument("--intake-plans", action="store_true",
                    help="original pipeline: the intake call writes the sub-queries (no supervisor route step)")
    ap.add_argument("--no-warm", action="store_true", help="don't preload the Ollama models and reranker at start-up")
    ap.add_argument("--company-name")
    ap.add_argument("--uen")
    ap.add_argument("--forget", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.quantize_rerank:
        os.environ["RERANK_QUANTIZE"] = "1"      # read by src.config, imported with the retriever below

    store = UserStore()
    if args.forget:
        store.forget(args.user)
        print(f"Deleted stored data for {args.user}")
        return
    if args.company_name or args.uen:
        store.set_company_info(args.user, company_name=args.company_name, uen=args.uen)

    sup = build_supervisor(args.max_steps, store=store, k=args.k, max_chunks=args.max_chunks,
                           max_queries=args.max_queries, candidates=args.candidates,
                           audit_min_score=args.audit_min_score, intake_plans=args.intake_plans)
    pin_ollama_keep_alive()                      # the decomposer's own ollama.chat calls keep their model loaded too
    if not args.no_warm:
        print("Loading Ollama models ...")
        for note in warm_models([os.environ.get("INTAKE_MODEL", "qwen3.5:4b"),
                                 os.environ.get("PLANNER_MODEL", os.environ.get("INTAKE_MODEL", "qwen3.5:4b")),
                                 MODEL, AUDITOR_MODEL]):
            print(f"  {note}")
        print("Loading the reranker ...")         # otherwise it loads inside the first question's retrieval time
        print(f"  {sup.retrieval.warm()}")
    messages = []
    while True:
        q = input("you> ").strip()
        if not q:
            break
        messages.append({"role": "user", "content": q})
        out = sup.handle_turn(args.user, messages)
        print(f"\nbot> {out.message}\n")
        if args.trace:
            route = f" route={out.route}" if out.route else ""
            print("    [" + out.status + route + "] " + "\n    ".join(out.trace) + "\n")
        messages.append({"role": "assistant", "content": out.message})
        if out.status != "clarify":
            messages = []          # a finished question starts a fresh conversation; the profile persists


if __name__ == "__main__":
    main()