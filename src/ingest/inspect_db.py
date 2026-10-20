"""
Shows stored document chunks and metadata inside ChromaDB.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb

from src.config import CHROMA_PATH, COLLECTION

parser = argparse.ArgumentParser(description="Print the chunks stored in ChromaDB (whole collection by default)")
parser.add_argument("--source", help="only show chunks from this PDF filename")
parser.add_argument("--limit", type=int, default=None, help="show at most N chunks (default: all)")
parser.add_argument("--full", action="store_true", help="print the full chunk text instead of a 300-character preview")
parser.add_argument("--summary", action="store_true", help="only print the per-document summary")
args = parser.parse_args()

collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION)
print(f"Collection '{COLLECTION}': {collection.count()} chunks, metadata {collection.metadata}")

res = collection.get(where={"source": args.source} if args.source else None, limit=args.limit)
rows = sorted(zip(res["ids"], res["documents"], res["metadatas"]),
              key=lambda r: (r[2]["source"], int(r[0].rsplit(":", 1)[1])))   # document order, numeric chunk index

print("\nPer document:")
per_doc = Counter(meta["source"] for _, _, meta in rows)
for source, n in per_doc.items():
    tokens = [m["n_tokens"] for _, _, m in rows if m["source"] == source]
    print(f"  {source}: {n} chunks, tokens min/avg/max {min(tokens)}/{round(sum(tokens) / n)}/{max(tokens)}")

if not args.summary:
    for id_, doc, meta in rows:
        print(f"\n# {id_} | pages {meta['pages']} | {meta['n_tokens']} tok | {meta['headings']}")
        print(doc if args.full else doc[:300].replace("\n", " "))
