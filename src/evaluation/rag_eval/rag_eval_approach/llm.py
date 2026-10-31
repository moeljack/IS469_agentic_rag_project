"""One callable shape for every model the evaluation uses: llm(messages, schema=None) -> str.

This is the shape the supervisor already uses (supervisor.default_llm), so any of these can replace the supervisor's
`llm` in Stage 3, write naive answers, or act as the judge. `messages` are chat dicts that may start with a
{"role": "system"} message; `schema` is an optional JSON schema the reply must follow (the reply is then JSON text).

    ollama      local model via the supervisor's own default_llm (temperature 0, thinking off)
    anthropic   Claude via the Anthropic SDK        (llm_anthropic.py; pip install anthropic)
    openai      OpenAI chat completions             (llm_openai.py;    pip install openai)
    google      Gemini via the google-genai SDK     (llm_google.py;    pip install google-genai)

Each provider lives in its own module and is imported only when used, so a missing SDK only matters for the runs that
need it. Token counts go to tracking.USAGE.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "supervisor"))   # supervisor.py is imported as a top-level module, as harness.py does


def split_system(messages):
    """(system text, the other messages): APIs that take the system prompt as a separate field need this."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def resolve_model(provider, model):
    """`model: default` on an Ollama provider means the project's SUPERVISOR_MODEL in src/config.py, so the evaluation
    runs the same model as the CLI and the app. Any other value is returned unchanged."""
    if provider == "ollama" and str(model) == "default":
        from src.config import SUPERVISOR_MODEL
        return SUPERVISOR_MODEL
    return model


def make_llm(provider, model, **options):
    model = resolve_model(provider, model)
    if not model or str(model).startswith("REPLACE"):
        raise ValueError(f"set an exact model id for provider {provider!r} in the config (got {model!r})")
    from src.evaluation.rag_eval.rag_eval_approach.env import has_key, load_env
    load_env()
    if not has_key(provider):
        raise RuntimeError(f"no API key for {provider!r}: add it to .env (see .env.example) or export it")
    if provider == "ollama":
        import supervisor
        from src.evaluation.rag_eval.rag_eval_approach.tracking import track_ollama
        track_ollama()
        return supervisor.default_llm(model)
    if provider == "anthropic":
        from src.evaluation.rag_eval.rag_eval_approach.llm_anthropic import anthropic_llm
        return anthropic_llm(model, **options)
    if provider == "openai":
        from src.evaluation.rag_eval.rag_eval_approach.llm_openai import openai_llm
        return openai_llm(model, **options)
    if provider == "groq":   # OpenAI-compatible endpoint; hosts open-weight models such as openai/gpt-oss-120b
        from src.evaluation.rag_eval.rag_eval_approach.llm_openai import openai_llm
        return openai_llm(model, base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY", **options)
    if provider == "google":
        from src.evaluation.rag_eval.rag_eval_approach.llm_google import google_llm
        return google_llm(model, **options)
    raise ValueError(f"unknown provider {provider!r}; choose ollama, anthropic, openai, groq or google")
