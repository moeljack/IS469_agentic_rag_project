"""Wire the real components together and chat in the terminal.

    python src/supervisor/harness.py --user acme            # interactive
    python src/supervisor/harness.py --user acme --trace    # also print what each agent did
    python src/supervisor/harness.py --user acme --forget   # delete everything stored for this user

Company details can be set once:  --company-name "Acme Pte Ltd" --uen 201912345A

Speed (all optional): --k, --max-chunks, --max-queries and --candidates shrink the retrieval work per question;
--no-warm skips preloading the Ollama models at start-up. --trace ends every answer with a per-stage timing line.

Pipeline:  decomposer (qwen3.5:4b)     ->  Supervisor (qwen3.5:4b)
                                              |-- RetrievalAgent -> retriever.Retriever (ChromaDB + BM25 + reranker)
                                              |-- Auditor        -> auditor.AuditorAgent (phi3.5:8b, ollama client)
                                              '-- UserStore (ChromaDB profile + pending clarifications)
Needs: ollama running with qwen3.5:4b and phi3.5:8b pulled (all models use the `ollama` Python client), and the
ChromaDB collection built by the ingest scripts.
Models: SUPERVISOR_MODEL / INTAKE_MODEL default to qwen3.5:4b; AUDITOR_MODEL defaults to phi3.5:8b.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.retriever import Retriever
from supervisor import (
    AUDITOR_MODEL, DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES, MODEL, Auditor, RetrievalAgent, Supervisor,
    pin_ollama_keep_alive, warm_models,
)
from src.retrieval.user_store import UserStore


def build_supervisor(max_steps: int = 4, tools: dict = None, store: UserStore = None, k: int = DEFAULT_K,
                     max_chunks: int = DEFAULT_MAX_CHUNKS, max_queries: int = DEFAULT_MAX_QUERIES,
                     candidates: int = None) -> Supervisor:
    """The real stack. Retriever and AuditorAgent are wrapped by the adapters in supervisor.py, which give them the
    .run() / .audit() interface the supervisor loop uses. The decomposer is the Supervisor's default planner."""
    return Supervisor(store=store or UserStore(),
                      retrieval=RetrievalAgent(Retriever(), k=k, max_chunks=max_chunks, candidates=candidates),
                      auditor=Auditor(),                         # AuditorAgent is created on first audit
                      tools=tools, max_steps=max_steps, max_queries=max_queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="user id (one profile per id)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4, help="max supervisor actions before it must answer")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="chunks kept per query after reranking")
    ap.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS, help="chunks the auditor judges per search")
    ap.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES, help="queries run per search")
    ap.add_argument("--candidates", type=int, default=None,
                    help="candidates the reranker scores per query (default: CANDIDATES in src/config.py); lower = faster")
    ap.add_argument("--no-warm", action="store_true", help="don't preload the Ollama models at start-up")
    ap.add_argument("--company-name")
    ap.add_argument("--uen")
    ap.add_argument("--forget", action="store_true")
    args = ap.parse_args()

    store = UserStore()
    if args.forget:
        store.forget(args.user)
        print(f"Deleted stored data for {args.user}")
        return
    if args.company_name or args.uen:
        store.set_company_info(args.user, company_name=args.company_name, uen=args.uen)

    sup = build_supervisor(args.max_steps, store=store, k=args.k, max_chunks=args.max_chunks,
                           max_queries=args.max_queries, candidates=args.candidates)
    pin_ollama_keep_alive()                      # the decomposer's own ollama.chat call keeps its model loaded too
    if not args.no_warm:
        print("Loading Ollama models ...")
        for note in warm_models([os.environ.get("INTAKE_MODEL", "qwen3.5:4b"), MODEL, AUDITOR_MODEL]):
            print(f"  {note}")
    messages = []
    while True:
        q = input("you> ").strip()
        if not q:
            break
        messages.append({"role": "user", "content": q})
        out = sup.handle_turn(args.user, messages)
        print(f"\nbot> {out.message}\n")
        if args.trace:
            print("    [" + out.status + "] " + "\n    ".join(out.trace) + "\n")
        messages.append({"role": "assistant", "content": out.message})
        if out.status != "clarify":
            messages = []          # a finished question starts a fresh conversation; the profile persists


if __name__ == "__main__":
    main()