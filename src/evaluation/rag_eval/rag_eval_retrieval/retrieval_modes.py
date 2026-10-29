"""The five retrieval configurations of the evaluation plan (Stage 1): R0-R4.

Retriever.search() always runs hybrid search + rerank (R4). The other configurations are built here from its stage
methods, so the production search path stays unchanged:

    dense          R0  vector search only
    bm25           R1  BM25 only
    hybrid         R2  vector + BM25 merged by reciprocal rank fusion, no rerank
    dense_rerank   R3  vector search, then the cross-encoder reranker
    hybrid_rerank  R4  Retriever.search() as it is

ModeRetriever.search() keeps Retriever.search()'s signature and hit format, so the agent's RetrievalAgent can use it
unchanged. `score` is the reranker's 0-1 relevance in the reranked modes; otherwise it is the vector cosine, the BM25
score or the RRF score, which are on different scales (so `min_score` and the agent's audit cutoff only make sense with
a reranked mode).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root, so `src` is importable

from src.config import CANDIDATES
from src.retrieval.retriever import RRF_K, Retriever

MODES = {
    "dense": "R0 dense",
    "bm25": "R1 BM25",
    "hybrid": "R2 hybrid",
    "dense_rerank": "R3 dense + rerank",
    "hybrid_rerank": "R4 hybrid + rerank",
}
RERANKED = {"dense_rerank", "hybrid_rerank"}


def _rrf(*rankings):
    """Reciprocal rank fusion scores, the same formula Retriever._candidates() uses to order the fused list."""
    fused = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            fused[id_] = fused.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
    return {id_: round(s, 5) for id_, s in fused.items()}


class ModeRetriever(Retriever):
    """A Retriever whose search() runs one of the five configurations. Change `mode` freely; the models stay loaded."""

    def __init__(self, mode="hybrid_rerank"):
        super().__init__()
        self.mode = mode

    def search(self, query, k=5, source=None, min_score=None, candidates=CANDIDATES):
        if self.mode not in MODES:
            raise ValueError(f"unknown retrieval mode {self.mode!r}; choose from {list(MODES)}")
        if self.mode == "hybrid_rerank":
            return super().search(query, k=k, source=source, min_score=min_score, candidates=candidates)

        n = max(candidates, k)   # candidate pool for the modes that rerank or fuse
        if self.mode == "bm25":
            vec, bm25 = {}, self._bm25_ranking(query, k, source)
            order, scores = list(bm25), bm25
        else:
            vector = self._encode_query(query)
            if self.mode == "dense":
                vec, bm25 = self._vector_ranking(vector, k, source), {}
                order, scores = list(vec), vec
            elif self.mode == "dense_rerank":
                vec, bm25 = self._vector_ranking(vector, n, source), {}
                rerank_scores = self._rerank(query, list(vec))
                order = sorted(vec, key=rerank_scores.get, reverse=True)
                scores = {i: round(rerank_scores[i], 4) for i in order}
            else:   # hybrid
                order, vec, bm25 = self._candidates(query, vector, n, source)
                scores = _rrf(vec, bm25)
        hits = self._hits(order[:k], scores, vec, bm25)
        return [h for h in hits if min_score is None or h["score"] >= min_score]
