import json
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

EMBED = "Qwen/Qwen3-Embedding-0.6B"
tok = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBED), max_tokens=512)
chunker = HybridChunker(tokenizer=tok, merge_peers=True)

ROOT = Path(__file__).resolve().parents[2]   # src/ingest/ingest.py -> project root
pdf = ROOT / "data" / "manual" / "etaxguide_Income-Tax-Treatment-of-REIT-Exchange-Traded-Funds.pdf"

doc = DocumentConverter().convert(pdf).document

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        embed_text = chunker.contextualize(chunk=ch)       # heading path + text -> embed this
        record = {
            "id": i,
            "text": ch.text,                                # raw chunk text
            "embed_text": embed_text,
            "headings": " > ".join(ch.meta.headings or []),  # store as metadata
            "pages": sorted({p.page_no for it in ch.meta.doc_items for p in it.prov}),  # PDFs only
            "n_tokens": tok.count_tokens(embed_text),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote chunks to {out_dir / 'chunks.jsonl'}")
