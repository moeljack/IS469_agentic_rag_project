import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.retrieval.retriever import Retriever

# ---- edit these, then run:  python src/retrieval/query.py ----
QUERIES = [
    "How is the trustee taxed on non-taxable income?",
    "What is the maximum IBNR percentage for motor insurance?",
]
K = 5               # chunks to return per question
SOURCE = None       # e.g. "etaxguide_....pdf" to search one document only; None searches all
MIN_SCORE = 0.5    # drop weak matches. With RERANK on this is the reranker's 0-1 relevance; None keeps all K
HYBRID = True       # also use BM25 keyword search (exact terms like "Annex 2B") next to vector search
RERANK = True       # re-score the candidates with the cross-encoder: slower, usually more accurate
FULL_TEXT = False   # True prints the whole chunk instead of a 300-character preview
# --------------------------------------------------------------

retriever = Retriever()   # loads the model once for all queries

for query in QUERIES:
    print(f"\n{'=' * 80}\n{query}\n{'=' * 80}")
    hits = retriever.search(query, k=K, source=SOURCE, min_score=MIN_SCORE, hybrid=HYBRID, rerank=RERANK)
    if not hits:
        print("No results.")
    for rank, h in enumerate(hits, 1):
        print(f"\n[{rank}] score {h['score']:.3f} | {h['id']} | pages {h['pages']}")
        print(f"    {h['headings'] or '(no heading)'}")
        print("    " + (h["text"] if FULL_TEXT else h["text"][:300]).replace("\n", "\n    "))
