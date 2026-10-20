"""
tests and inspects retrieval and semantic reranking results against ChromaDB
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.retrieval.retriever import Retriever

# ---- edit these, then run:  python src/retrieval/query.py ----
QUERIES = [
    "How is the trustee taxed on non-taxable income?",
    "What is the maximum IBNR percentage for motor insurance?",
    "What is the corporate income tax rate?",
]
K = 5               # chunks to return per question
SOURCE = None       # e.g. "etaxguide_....pdf" or a page URL to search one document only
MIN_SCORE = None    # drop weak matches: the reranker's 0-1 relevance, e.g. 0.5; None keeps all K
FULL_TEXT = False   # True prints the whole chunk instead of a 300-character preview
# --------------------------------------------------------------

retriever = Retriever()   # loads the models once for all queries

for query in QUERIES:
    print(f"\n{'=' * 80}\n{query}\n{'=' * 80}")
    hits = retriever.search(query, k=K, source=SOURCE, min_score=MIN_SCORE)
    if not hits:
        print("No results.")
    for rank, h in enumerate(hits, 1):
        where = h["source"] if h["source_type"] == "webpage" else f"{h['id']} | pages {h['pages']}"
        print(f"\n[{rank}] score {h['score']:.3f} | {h['source_type']} | {where}")
        print(f"    {h['headings'] or '(no heading)'}")
        print("    " + (h["text"] if FULL_TEXT else h["text"][:300]).replace("\n", "\n    "))
