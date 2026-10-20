"""
Executes a multi-stage search across indexed document chunks

retrieval.search() returns in a json format
| Key | Description |
| :--- | :--- |
| `id` | Unique chunk ID (`file:index`). |
| `score` | Cross-encoder relevance (`0.0–1.0`). |
| `text` | Clean passage text for LLM context. |
| `headings` | Hierarchical section path. |
| `source` / `source_type` | File/URL and format (`pdf`/`webpage`). |
| `pages` | Original PDF page numbers. |
| `vector_score` / `bm25_score` | Dense cosine similarity and sparse lexical match scores. |
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

import chromadb
import numpy as np
import torch
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    CANDIDATES, CHROMA_PATH, COLLECTION, EMBED_MODEL, EMBED_MODEL_DIR, RERANK_MAX_LENGTH, RERANK_MODEL_DIR,
    RERANK_QUANTIZE,
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
    """Finds the chunks that best answer a question, from the web pages and the PDFs.

    Each search runs up to three stages:
      1. candidates: vector search (meaning) and BM25 (exact words like "Annex 2B" or "section 45G"),
         merged by reciprocal rank fusion;
      2. a long-context cross-encoder reranker that reads question and chunk together;
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
    def _where(source):
        return {"source": source} if source else None

    def _encode_query(self, query):
        return self.model.encode(query, prompt_name="query", normalize_embeddings=True)   # asymmetric: query prompt

    def _vector_ranking(self, vector, n, source):
        res = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=n, where=self._where(source), include=["distances"],
        )
        return {id_: round(1 - dist, 4) for id_, dist in zip(res["ids"][0], res["distances"][0])}   # cosine similarity

    def _bm25_ranking(self, query, n, source):
        scores = self._bm25.get_scores(_tokens(query))
        order = np.argsort(-scores)
        ranking = {}
        for i in order:
            id_ = self._ids[i]
            if scores[i] <= 0 or len(ranking) >= n:
                break
            if source is None or self._chunks[id_]["source"] == source:
                ranking[id_] = float(scores[i])
        return ranking

    def _candidates(self, query, vector, n, source):
        """The n best chunks by the fast first stage: (ordered ids, vector scores, bm25 scores)."""
        vec = self._vector_ranking(vector, n, source)
        bm25 = self._bm25_ranking(query, n, source)
        fused = {}   # reciprocal rank fusion: chunks ranked high in either list rise to the top
        for ranking in (vec, bm25):
            for rank, id_ in enumerate(ranking):
                fused[id_] = fused.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)[:n], vec, bm25

    def _rerank(self, query, ids):
        if self._reranker is None:
            from src.ingest.download_models import ensure_reranker
            ensure_reranker()   # first use only: downloads into ./models
            self._reranker = CrossEncoder(str(RERANK_MODEL_DIR), max_length=RERANK_MAX_LENGTH)
            if RERANK_QUANTIZE:   # int8 weights for the linear layers: faster on CPU, scores change only slightly
                self._reranker.model = torch.quantization.quantize_dynamic(
                    self._reranker.model, {torch.nn.Linear}, dtype=torch.qint8)
        pairs = [(query, _rerank_text(self._chunks[i]["headings"], self._chunks[i]["text"])) for i in ids]
        return dict(zip(ids, self._reranker.predict(pairs, batch_size=8).tolist()))   # 0..1 relevance

    def _hits(self, ids, scores, vec, bm25):
        return [{"id": i, "score": scores[i], **self._chunks[i], "vector_score": vec.get(i), "bm25_score": bm25.get(i)}
                for i in ids]

    # ---- public ----
    def search(self, query, k=5, source=None, min_score=None, candidates=CANDIDATES):
        """Top-k chunks for `query`, best first. Web pages and PDFs are searched together; each hit has a `source_type`.

        source:     only search this document (PDF filename or page URL).
        candidates: size of the candidate pool handed to the reranker (the pool is at least k).
        min_score:  drop hits whose `score` is below this. `score` is the reranker's 0-1 relevance.
        """
        n = max(candidates, k)
        order, vec, bm25 = self._candidates(query, self._encode_query(query), n, source)
        rerank_scores = self._rerank(query, order)
        order = sorted(order, key=rerank_scores.get, reverse=True)
        scores = {i: round(rerank_scores[i], 4) for i in order}
        hits = self._hits(order[:k], scores, vec, bm25)
        return [h for h in hits if min_score is None or h["score"] >= min_score]
