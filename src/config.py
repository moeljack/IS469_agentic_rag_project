from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # src/config.py -> project root

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"    # HF repo id; recorded in the collection to detect model changes
MAX_TOKENS = 512                             # chunk size limit (must fit the embedder's window)

MANUAL_DIR = ROOT / "data" / "manual"
CHROMA_PATH = ROOT / "chroma_db"
COLLECTION = "manuals"

# Local model copies (git-ignored). Filled by src/ingest/download_models.py using
# snapshot_download(local_dir=...), which avoids the symlinks that fail on Windows.
MODELS_DIR = ROOT / "models"
DOCLING_ARTIFACTS = MODELS_DIR / "docling"
EMBED_MODEL_DIR = MODELS_DIR / EMBED_MODEL.split("/")[-1]
