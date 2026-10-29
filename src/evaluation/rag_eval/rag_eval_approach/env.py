"""Load API keys from the repo's `.env` (gitignored) into os.environ, without needing python-dotenv.

Lines are KEY=value; blank lines and # comments are skipped; surrounding quotes are removed. A variable that is already
set in the real environment wins over the file, so the cluster can export its keys instead of keeping a file.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
KEYS = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",), "groq": ("GROQ_API_KEY",), "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def load_env(path=ROOT / ".env"):
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def has_key(provider):
    """True if the provider's API key is set (ollama needs none)."""
    return provider == "ollama" or any(os.environ.get(k) for k in KEYS.get(provider, ()))
