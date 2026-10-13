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

Inspecting with a GUI: after ingest finishes, run `chroma run --path chroma_db` and connect the GUI to
`http://localhost:8000`; press Ctrl+C when done. `ingest.py` holds port 8000 while it runs
(see `src/ingest/chroma_guard.py`), so it refuses to start if the server is up, and the server refuses to start
if ingest is running. If you run the server on another port, set `CHROMA_SERVER_PORT` in `src/config.py`.

Notes:
- OCR is off, so scanned (image-only) PDFs produce no text and are reported as failed.
- One failing PDF does not stop the run; failures are listed at the end.
- `models/`, `chroma_db/` and the virtual environment are git-ignored.

## Troubleshooting

- `WinError 1314` (symlink privilege) from Hugging Face: turn on Windows Developer Mode
  (Settings > System > For developers). The `models/` download avoids this, but other
  Hugging Face downloads may still hit it.
- `No matching distribution found for torch`: the venv uses an unsupported Python version; recreate it with 3.12.
