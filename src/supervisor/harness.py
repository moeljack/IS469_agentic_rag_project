"""Wire the real components together and chat in the terminal.

    python src/supervisor/harness.py --user acme            # interactive
    python src/supervisor/harness.py --user acme --trace    # also print what each agent did
    python src/supervisor/harness.py --user acme --forget   # delete everything stored for this user

Company details can be set once:  --company-name "Acme Pte Ltd" --uen 201912345A

Pipeline:  intake_planner (qwen3.5:4b)  ->  Supervisor (qwen3.5:4b)
                                              |-- RetrievalAgent -> retriever.Retriever (ChromaDB + BM25 + reranker)
                                              |-- Auditor        -> auditor.AuditorAgent (phi3.5:8b, ollama client)
                                              '-- UserStore (ChromaDB profile + pending clarifications)
Needs: ollama running with qwen3.5:4b and phi3.5:8b pulled (all models use the `ollama` Python client), and the
ChromaDB collection built by the ingest scripts.
Models: SUPERVISOR_MODEL / INTAKE_MODEL default to qwen3.5:4b; AUDITOR_MODEL defaults to phi3.5:8b.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.retriever import Retriever
from supervisor import Auditor, RetrievalAgent, Supervisor
from src.retrieval.user_store import UserStore


def build_supervisor(max_steps: int = 4, tools: dict = None, store: UserStore = None) -> Supervisor:
    """The real stack. Retriever and AuditorAgent are wrapped by the adapters in supervisor.py, which give them the
    .run() / .audit() interface the supervisor loop uses. The intake planner is the Supervisor's default."""
    return Supervisor(store=store or UserStore(),
                      retrieval=RetrievalAgent(Retriever()),     # loads embedder + BM25 index once
                      auditor=Auditor(),                         # AuditorAgent is created on first audit
                      tools=tools, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="user id (one profile per id)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4, help="max supervisor actions before it must answer")
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

    sup = build_supervisor(args.max_steps, store=store)
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