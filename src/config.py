import os
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

# Retrieval: hybrid search (vector + BM25 keyword) then cross-encoder reranking
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")   # reads up to 8192 tokens; set the RERANK_MODEL environment variable to try another
RERANK_MODEL_DIR = MODELS_DIR / RERANK_MODEL.split("/")[-1]
RERANK_QUANTIZE = os.environ.get("RERANK_QUANTIZE", "0") == "1"   # int8 weights on CPU: faster reranking, tiny accuracy change (see evaluate.py)
RERANK_MAX_LENGTH = 1024   # tokens of (question + chunk) the reranker reads; our longest chunks need about 620
CANDIDATES = int(os.environ.get("CANDIDATES", 10))   # chunks pulled from each ranking and handed to the reranker

# The local LLMs (all run through Ollama). One place to change them; each can still be overridden by its environment
# variable. The supervisor and the evaluation's `generator: model: default` read SUPERVISOR_MODEL, so the CLI, the
# web app and the evaluation run the same models unless a config names another one on purpose.
LLM_MODEL = "qwen3.5:9b"                                                        # tag to `ollama pull`
INTAKE_MODEL = os.environ.get("INTAKE_MODEL", LLM_MODEL)                        # intake gate: domain, intent, entities
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", INTAKE_MODEL)                   # follow-up search queries
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", INTAKE_MODEL)             # route, decide and answer calls
AUDITOR_MODEL = os.environ.get("AUDITOR_MODEL", LLM_MODEL)                      # judges each retrieved chunk

# Scraped web pages (JSON lines: url, title, text, retrieved_at, source, source_type)
RAW_DIR = ROOT / "data" / "raw"
