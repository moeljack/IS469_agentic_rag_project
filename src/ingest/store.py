"""
Shared by the PDF and web ingest scripts: open the collection and refuse to mix embedding models.

opens, initialises, and verifies a ChromaDB vector collection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from chromadb.errors import NotFoundError

from src.config import COLLECTION, EMBED_MODEL


def open_collection(client, embedder, reset=False):
    """Open the collection, refusing to mix vectors from different embedding models."""
    expected = {"embed_model": EMBED_MODEL, "embed_dim": embedder.get_sentence_embedding_dimension()}
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except NotFoundError:
            pass
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine", **expected})
    stored = {k: (collection.metadata or {}).get(k) for k in expected}
    if stored != expected:   # Chroma silently keeps the old metadata of an existing collection
        if collection.count() == 0:
            client.delete_collection(COLLECTION)
            return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", **expected})
        sys.exit(
            f"Collection '{COLLECTION}' was built with {stored}, but config now says {expected}.\n"
            "Mixing models would quietly degrade retrieval. Re-run with --reset to rebuild everything "
            "(or restore EMBED_MODEL in src/config.py)."
        )
    return collection
