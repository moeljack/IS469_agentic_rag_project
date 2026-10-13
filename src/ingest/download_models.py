"""Download the models into ./models so nothing depends on the symlink-based HF cache.

ingest.py calls ensure_models() automatically on the first run. To download ahead of time
(or after changing EMBED_MODEL):  python src/ingest/download_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.config import DOCLING_ARTIFACTS, EMBED_MODEL, EMBED_MODEL_DIR

DONE_MARKER = ".download_complete"   # written last, so an interrupted download is retried, not trusted


def _is_done(path):
    return (path / DONE_MARKER).exists()


def ensure_models():
    """Download whatever is missing; do nothing if both models are already in ./models."""
    from docling.utils.model_downloader import download_models   # imported lazily: heavy
    from huggingface_hub import snapshot_download

    if not _is_done(DOCLING_ARTIFACTS):
        print(f"Downloading Docling layout + table models to {DOCLING_ARTIFACTS} ...")
        # OCR, code/formula enrichment and picture classification are off in ingest.py, so skip them.
        download_models(
            output_dir=DOCLING_ARTIFACTS,
            progress=True,
            with_layout=True,
            with_tableformer=True,
            with_code_formula=False,
            with_picture_classifier=False,
            with_rapidocr=False,
        )
        (DOCLING_ARTIFACTS / DONE_MARKER).touch()

    if not _is_done(EMBED_MODEL_DIR):
        print(f"Downloading {EMBED_MODEL} (about 1.2 GB) to {EMBED_MODEL_DIR} ...")
        snapshot_download(repo_id=EMBED_MODEL, local_dir=EMBED_MODEL_DIR)
        (EMBED_MODEL_DIR / DONE_MARKER).touch()


if __name__ == "__main__":
    ensure_models()
    print("Models ready.")
