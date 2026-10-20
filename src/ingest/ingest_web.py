"""
Chunk, embed and store scraped web pages (data/raw/*.jsonl) in the same ChromaDB collection as the PDFs.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config import CHROMA_PATH, COLLECTION, EMBED_MODEL_DIR, MAX_TOKENS, RAW_DIR
from src.ingest.download_models import ensure_models
from src.ingest.store import open_collection
from src.ingest.web_chunking import chunk_page, clean_text, clean_title

BATCH_SIZE = 16


def load_pages(raw_dir):
    """Records from every *.jsonl in raw_dir that are web pages, with exact duplicates merged.

    Landing pages often repeat their first child page word for word. We keep the most specific URL and
    remember the others in `also_at`, so the same text is never stored twice.
    """
    pages, bad = [], 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if record.get("source_type") == "webpage" and record.get("url") and record.get("text", "").strip():
                pages.append(record)
    if bad:
        print(f"WARNING: skipped {bad} unreadable line(s)")

    groups = {}
    for record in pages:
        key = hashlib.md5(re.sub(r"\s+", " ", record["text"]).strip().encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(record)
    unique = []
    for group in groups.values():
        group.sort(key=lambda r: len(r["url"]), reverse=True)   # longest (most specific) URL first
        keep = dict(group[0])
        keep["also_at"] = [r["url"] for r in group[1:]]
        unique.append(keep)
    return unique, len(pages) - len(unique)


def ingest_page(record, tokenizer, embedder, collection):
    title = clean_title(record.get("title", ""), record.get("source", ""))
    count = lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    chunks = chunk_page(title, clean_text(record["text"]), count, MAX_TOKENS)
    if not chunks:
        raise RuntimeError("no text left after cleaning")

    url = record["url"]
    prefix = f"web:{hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]}"
    ids = [f"{prefix}:{i}" for i in range(len(chunks))]
    embed_texts = [f"{c['headings']}\n{c['text']}" for c in chunks]          # heading path + text -> embed this
    metadatas = []
    for c in chunks:
        meta = {                                            # same keys as the PDF chunks, so searches treat both alike
            "source": url,                                  # unique per page: used to skip / replace a page
            "source_type": "webpage",
            "headings": c["headings"],
            "pages": "",
            "page_start": -1,
            "n_tokens": c["n_tokens"],
            "title": title,
            "publisher": record.get("source", ""),
            "retrieved_at": record.get("retrieved_at", ""),
        }
        if record["also_at"]:
            meta["also_at"] = ",".join(record["also_at"])
        metadatas.append(meta)

    embeddings = embedder.encode(embed_texts, batch_size=BATCH_SIZE, normalize_embeddings=True)
    collection.delete(where={"source": url})                # drop the old version of this page
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=[c["text"] for c in chunks],
                   metadatas=metadatas)
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Chunk, embed and store scraped web pages in ChromaDB")
    parser.add_argument("--force", action="store_true", help="re-ingest pages even if they are unchanged")
    args = parser.parse_args()

    ensure_models()
    pages, merged = load_pages(RAW_DIR)
    if not pages:
        sys.exit(f"No web pages found in {RAW_DIR}/*.jsonl")
    print(f"{len(pages)} unique pages ({merged} exact duplicate(s) merged)")

    embedder = SentenceTransformer(str(EMBED_MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(EMBED_MODEL_DIR))
    collection = open_collection(chromadb.PersistentClient(path=str(CHROMA_PATH)), embedder)

    ingested, skipped, failed = 0, 0, []
    for record in pages:
        url = record["url"]
        try:
            stored = collection.get(where={"source": url}, limit=1, include=["metadatas"])["metadatas"]
            if not args.force and stored and stored[0].get("retrieved_at") == record.get("retrieved_at"):
                skipped += 1                                # unchanged since the last ingest
                continue
            n = ingest_page(record, tokenizer, embedder, collection)
            print(f"  {n:3d} chunks | {url.split('/taxes/')[-1][:80]}")
            ingested += 1
        except Exception as e:                              # one bad page must not stop the run
            print(f"  FAILED {url}: {type(e).__name__}: {e}")
            failed.append((url, f"{type(e).__name__}: {e}"))

    print(f"\nIngested {ingested}, skipped {skipped} unchanged, failed {len(failed)}. "
          f"Collection '{COLLECTION}' holds {collection.count()} chunks")
    if failed:
        print("Failed pages:")
        for url, err in failed:
            print(f"  - {url}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
