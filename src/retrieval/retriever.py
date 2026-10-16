import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

import chromadb
import numpy as np
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    CANDIDATES, CHROMA_PATH, COLLECTION, EMBED_MODEL, EMBED_MODEL_DIR, RERANK_MODEL_DIR, WEB_FIRST_MIN_SCORE,
)

RRF_K = 60   # reciprocal-rank-fusion constant; 60 is the standard default


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())   # "section 45G" -> section, 45g; "35(15)" -> 35, 15


def _rerank_text(headings, text):
    """Heading path + text, with the Markdown table padding squeezed out so it doesn't eat the token budget."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-{3,}", "---", text)
    return f"{headings}\n{text}" if headings else text


class Retriever:
    """Finds the chunks that best answer a question.

    search() runs up to three stages:
      1. vector search (meaning) and, if hybrid, BM25 (exact words like "Annex 2B" or "section 45G"),
         merged by reciprocal rank fusion;
      2. a cross-encoder reranker that reads question and chunk together;
      3. the top k.

    Create it once and reuse it: loading the models takes a few seconds. The BM25 index is built from the
    collection at start-up, so create a new Retriever after re-ingesting.
    """

    def __init__(self):
        if not EMBED_MODEL_DIR.exists():
            raise FileNotFoundError(f"{EMBED_MODEL_DIR} not found. Run python src/ingest/ingest.py first.")
        try:
            self.collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION)
        except NotFoundError:
            raise RuntimeError(f"Collection '{COLLECTION}' not found. Run python src/ingest/ingest.py first.") from None

        built_with = (self.collection.metadata or {}).get("embed_model")
        if built_with != EMBED_MODEL:   # queries must live in the same vector space as the chunks
            raise RuntimeError(
                f"Collection was built with {built_with!r} but config says {EMBED_MODEL!r}. "
                "Run python src/ingest/ingest.py --reset to rebuild."
            )
        self.model = SentenceTransformer(str(EMBED_MODEL_DIR))

        data = self.collection.get(include=["documents", "metadatas"])   # whole collection, for BM25 and reranking
        self._ids = data["ids"]
        self._chunks = {
            id_: {"text": text, "source": meta["source"], "pages": meta["pages"], "headings": meta["headings"],
                  "source_type": meta.get("source_type", ""), "title": meta.get("title", "")}
            for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self._bm25 = BM25Okapi([_tokens(f"{self._chunks[i]['headings']} {self._chunks[i]['text']}") for i in self._ids])
        self._reranker = None   # loaded on first use

    # ---- stages ----
    @staticmethod
    def _where(source, source_type):
        conditions = [{"source": source}] if source else []
        if source_type:
            conditions.append({"source_type": source_type})
        return None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _vector_ranking(self, query, n, source, source_type):
        vector = self.model.encode(query, prompt_name="query", normalize_embeddings=True)   # asymmetric: query prompt
        res = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=n,
            where=self._where(source, source_type), include=["distances"],
        )
        return {id_: round(1 - dist, 4) for id_, dist in zip(res["ids"][0], res["distances"][0])}   # cosine similarity

    def _bm25_ranking(self, query, n, source, source_type):
        scores = self._bm25.get_scores(_tokens(query))
        order = np.argsort(-scores)
        ranking = {}
        for i in order:
            id_ = self._ids[i]
            if scores[i] <= 0 or len(ranking) >= n:
                break
            chunk = self._chunks[id_]
            if (source is None or chunk["source"] == source) and (source_type is None or chunk["source_type"] == source_type):
                ranking[id_] = float(scores[i])
        return ranking

    def _rerank(self, query, ids):
        if self._reranker is None:
            from src.ingest.download_models import ensure_reranker
            ensure_reranker()   # first use only: downloads into ./models
            self._reranker = CrossEncoder(str(RERANK_MODEL_DIR), max_length=512)
        pairs = [(query, _rerank_text(self._chunks[i]["headings"], self._chunks[i]["text"])) for i in ids]
        return dict(zip(ids, self._reranker.predict(pairs, batch_size=8).tolist()))   # 0..1 relevance

    # ---- public ----
    def search(self, query, k=5, source=None, min_score=None, hybrid=True, rerank=True, candidates=CANDIDATES,
               source_type=None):
        """Top-k chunks for `query`, best first.

        source:    only search this document (PDF filename or page URL).
        source_type: only search "pdf" or "webpage" documents.
        hybrid:    also use BM25 keyword search and merge it with the vector results.
        rerank:    re-score the candidates with the cross-encoder (slower, usually more accurate).
        min_score: drop hits whose final `score` is below this. `score` is the reranker's 0-1 relevance when
                   rerank=True, otherwise the vector cosine similarity (or the fused rank score if hybrid).
        """
        n = max(candidates, k)
        vec = self._vector_ranking(query, n, source, source_type)
        bm25 = self._bm25_ranking(query, n, source, source_type) if hybrid else {}

        if hybrid:   # reciprocal rank fusion: chunks ranked high in either list rise to the top
            fused = {}
            for ranking in (vec, bm25):
                for rank, id_ in enumerate(ranking):
                    fused[id_] = fused.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)[:n]
            scores = {i: round(fused[i], 4) for i in order}
        else:
            order, scores = list(vec)[:n], dict(list(vec.items())[:n])

        rerank_scores = {}
        if rerank:
            rerank_scores = self._rerank(query, order)
            order = sorted(order, key=rerank_scores.get, reverse=True)
            scores = {i: round(rerank_scores[i], 4) for i in order}

        hits = [
            {"id": i, "score": scores[i], **self._chunks[i],
             "vector_score": vec.get(i), "bm25_score": bm25.get(i)}
            for i in order[:k]
        ]
        return [h for h in hits if min_score is None or h["score"] >= min_score]

    def search_tiered(self, query, k=5, min_score=None, web_threshold=WEB_FIRST_MIN_SCORE, **options):
        """Web pages first; the PDFs only if the web pages don't answer the question.

        The web pages are searched on their own. If their best hit reaches `web_threshold` (the reranker's 0-1
        relevance), those hits are returned. Otherwise the PDFs are searched and their hits are returned, each marked
        `fallback=True` so the caller can tell the answer did not come from the web pages.
        Reranking is always on here, because the threshold is a reranker score.
        """
        options.pop("rerank", None)
        web = self.search(query, k=k, min_score=min_score, source_type="webpage", rerank=True, **options)
        if web and web[0]["score"] >= web_threshold:
            return [{**h, "fallback": False} for h in web]
        pdf = self.search(query, k=k, min_score=min_score, source_type="pdf", rerank=True, **options)
        return [{**h, "fallback": True} for h in pdf]
