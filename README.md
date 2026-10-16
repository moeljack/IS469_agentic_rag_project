# IS469_agentic_rag_project

## Setup (Windows)

Requires **Python 3.12** (3.11 also works). PyTorch does not publish wheels for very new Python versions.

```powershell
py -3.12 -m venv projenv
.\projenv\Scripts\Activate.ps1
python --version                     # should say 3.12.x
pip install -r requirements.txt
```

## Ingest documents

1. Put PDFs in `data/manual/`.
2. Run:
   ```powershell
   python src/ingest/ingest.py
   ```

On the **first run** the script downloads the models it needs into `models/`
(Docling layout/table models and `Qwen/Qwen3-Embedding-0.6B`, about 1.2 GB in total).
Later runs reuse them. To download ahead of time: `python src/ingest/download_models.py`.

What it does: PDF -> Docling -> chunks -> embeddings -> ChromaDB (`chroma_db/`, collection `manuals`).

| Command | Effect |
|---|---|
| `python src/ingest/ingest.py` | Ingest new PDFs, skip ones already stored |
| `python src/ingest/ingest.py --force` | Re-ingest every PDF |
| `python src/ingest/ingest.py --reset` | Wipe the collection and rebuild (required after changing `EMBED_MODEL`) |

### Scraped web pages

Put the scraper output (JSON lines with `url`, `title`, `text`, `retrieved_at`, `source`, `source_type: "webpage"`) in `data/raw/`, then run:

```powershell
python src/ingest/ingest_web.py            # new or re-scraped pages only (--force redoes everything)
```

Exact duplicate pages are merged, text is cleaned, and each page is split by section and table into chunks stored in
the same collection as the PDFs (`source` = the page URL, `source_type` = `webpage`).

## Searching

Edit the settings at the top of `src/retrieval/query.py`, then run `python src/retrieval/query.py`.
With `WEB_FIRST = True` the web pages are searched first and the PDFs are used only when no web page answers the
question (`WEB_FIRST_MIN_SCORE` in `src/config.py`). `python src/retrieval/evaluate.py` scores retrieval on
`src/retrieval/eval_questions.json` (add your own questions there).

Inspecting with a GUI: after ingest finishes, run `chroma run --path chroma_db` and connect the GUI to
`http://localhost:8000`. **Stop the server (Ctrl+C) before running `ingest.py` or `inspect_db.py`**: two processes
using `chroma_db/` at the same time can lock or corrupt it.

Notes:
- OCR is off, so scanned (image-only) PDFs produce no text and are reported as failed.
- One failing PDF does not stop the run; failures are listed at the end.
- `models/`, `chroma_db/` and the virtual environment are git-ignored.

## Troubleshooting

- `WinError 1314` (symlink privilege) from Hugging Face: turn on Windows Developer Mode
  (Settings > System > For developers). The `models/` download avoids this, but other
  Hugging Face downloads may still hit it.
- `No matching distribution found for torch`: the venv uses an unsupported Python version; recreate it with 3.12.
