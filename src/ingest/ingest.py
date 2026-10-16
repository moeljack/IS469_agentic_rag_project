import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

import chromadb
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config import (
    CHROMA_PATH, COLLECTION, DOCLING_ARTIFACTS, EMBED_MODEL_DIR, MANUAL_DIR, MAX_TOKENS,
)
from src.ingest.chunking import build_chunker, fix_structure
from src.ingest.download_models import ensure_models
from src.ingest.store import open_collection

BATCH_SIZE = 16


def build_converter():
    # OCR off (text PDFs only) and models read from a local folder, so no HF cache symlinks are needed
    opts = PdfPipelineOptions(artifacts_path=DOCLING_ARTIFACTS, do_ocr=False)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def ingest_pdf(pdf, converter, chunker, tok, embedder, collection, window):
    doc = fix_structure(converter.convert(pdf).document)

    ids, documents, embed_texts, metadatas = [], [], [], []
    oversized = 0
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        embed_text = chunker.contextualize(chunk=ch)       # heading path + text -> embed this
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in it.prov})  # PDFs only
        n_tokens = tok.count_tokens(embed_text)
        oversized += bool(window and n_tokens > window)
        ids.append(f"{pdf.stem}:{i}")
        documents.append(ch.text)                           # raw text the LLM will read/cite
        embed_texts.append(embed_text)
        metadatas.append({                                  # Chroma: no lists, no None
            "source": pdf.name,
            "source_type": "pdf",                           # lets web-scraped sources share the collection
            "headings": " > ".join(ch.meta.headings or []),
            "pages": ",".join(map(str, pages)),
            "page_start": pages[0] if pages else -1,
            "n_tokens": n_tokens,
        })

    if not ids:
        raise RuntimeError("no chunks produced (scanned PDF? OCR is turned off)")
    if oversized:
        print(f"  WARNING: {oversized} chunk(s) exceed the embedder window of {window} tokens and will be truncated")

    embeddings = embedder.encode(embed_texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                 show_progress_bar=True)

    collection.delete(where={"source": pdf.name})           # drop stale chunks on re-ingest
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Chunk, embed and store PDFs from data/manual in ChromaDB")
    parser.add_argument("--force", action="store_true", help="re-ingest documents already in the collection")
    parser.add_argument("--reset", action="store_true",
                        help="delete the whole collection first (needed after changing EMBED_MODEL); implies --force")
    args = parser.parse_args()
    force = args.force or args.reset

    ensure_models()   # first run only: downloads into ./models (no HF cache symlinks)

    pdfs = sorted(MANUAL_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {MANUAL_DIR}")

    embedder = SentenceTransformer(str(EMBED_MODEL_DIR))
    window = embedder.max_seq_length                       # tokens the embedder reads before truncating
    if window and MAX_TOKENS > window:
        sys.exit(f"MAX_TOKENS={MAX_TOKENS} is larger than the embedder window ({window}); lower it in src/config.py")

    tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(str(EMBED_MODEL_DIR)), max_tokens=MAX_TOKENS)
    chunker = build_chunker(tok)
    converter = build_converter()

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = open_collection(client, embedder, reset=args.reset)

    ingested, skipped, failed = [], [], []
    for pdf in pdfs:
        try:
            if not force and collection.get(where={"source": pdf.name}, limit=1)["ids"]:
                print(f"Skipping {pdf.name} (already ingested; use --force to redo)")
                skipped.append(pdf.name)
                continue
            print(f"Ingesting {pdf.name} ...")
            n = ingest_pdf(pdf, converter, chunker, tok, embedder, collection, window)
            print(f"  {n} chunks")
            ingested.append(pdf.name)
        except Exception as e:                              # one bad PDF must not stop the run
            print(f"  FAILED {pdf.name}: {type(e).__name__}: {e}")
            failed.append((pdf.name, f"{type(e).__name__}: {e}"))

    print(f"\nIngested {len(ingested)}, skipped {len(skipped)}, failed {len(failed)}. "
          f"Collection '{COLLECTION}' holds {collection.count()} chunks")
    if failed:
        print("Failed files:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
