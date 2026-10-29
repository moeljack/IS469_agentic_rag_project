"""Shared helpers for the dataset scripts: load every chunk from ChromaDB and find the chunks that contain a fact.

Matching is the same as src/evaluation/rag_eval/evaluate_rag.py: heading path + text, ignoring case, spacing and
apostrophe style.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))   # make `src` importable

import chromadb

from src.config import CHROMA_PATH, COLLECTION

DATASET = Path(__file__).resolve().parents[2] / "eval_questions.json"


def norm(text):
    return re.sub(r"\s+", " ", text).lower().replace("’", "'")


def load_chunks(db=CHROMA_PATH, collection=COLLECTION):
    data = chromadb.PersistentClient(path=str(db)).get_collection(collection).get(include=["documents", "metadatas"])
    return [{"id": id_, "hay": norm(f"{meta.get('headings', '')} {text}")}
            for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])]


def matching_ids(chunks, alternatives):
    """Ids of the chunks containing any of the alternative strings of one fact."""
    wanted = [norm(a) for a in alternatives]
    return sorted({c["id"] for c in chunks if any(w in c["hay"] for w in wanted)})
