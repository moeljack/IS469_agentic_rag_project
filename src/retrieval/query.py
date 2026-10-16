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
WEB_FIRST = True    # search the web pages first; use the PDFs only if the web pages do not answer (always reranks)
SOURCE = None       # e.g. "etaxguide_....pdf" or a page URL to search one document only (ignored if WEB_FIRST)
MIN_SCORE = 0.5     # drop weak matches. With reranking this is the reranker's 0-1 relevance; None keeps all K
HYBRID = True       # also use BM25 keyword search (exact terms like "Annex 2B") next to vector search
RERANK = True       # re-score the candidates with the cross-encoder: slower, more accurate (ignored if WEB_FIRST)
FULL_TEXT = False   # True prints the whole chunk instead of a 300-character preview
# --------------------------------------------------------------

retriever = Retriever()   # loads the models once for all queries

for query in QUERIES:
    print(f"\n{'=' * 80}\n{query}\n{'=' * 80}")
    if WEB_FIRST:
        hits = retriever.search_tiered(query, k=K, min_score=MIN_SCORE, hybrid=HYBRID)
        if hits:
            print("(answered from the web pages)" if not hits[0]["fallback"]
                  else "(not found on the web pages: showing PDF results)")
    else:
        hits = retriever.search(query, k=K, source=SOURCE, min_score=MIN_SCORE, hybrid=HYBRID, rerank=RERANK)
    if not hits:
        print("No results.")
    for rank, h in enumerate(hits, 1):
        where = h["source"] if h["source_type"] == "webpage" else f"{h['id']} | pages {h['pages']}"
        print(f"\n[{rank}] score {h['score']:.3f} | {h['source_type']} | {where}")
        print(f"    {h['headings'] or '(no heading)'}")
        print("    " + (h["text"] if FULL_TEXT else h["text"][:300]).replace("\n", "\n    "))
