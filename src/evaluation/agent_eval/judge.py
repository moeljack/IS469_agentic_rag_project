"""The fixed LLM judge for agent_eval's LLM-graded scores (RAGAS faithfulness and answer relevance).

Copied from rag_eval so agent_eval does not import it:
    load_env, has_key, KEYS   <- rag_eval/rag_eval_approach/env.py
    chat_model                <- rag_eval/rag_eval_approach/score_ragas.py
Added here: provider "ollama" in chat_model. rag_eval's judge.yaml names an Ollama model (gemma4:12b), but its
score_ragas.chat_model has no Ollama branch, so it could not run with that config.

The judge config is agent_eval/configs/judge.yaml (a copy of rag_eval's). Keep both on the same model if the two
reports are compared, and never change it between runs.
"""
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                              # repo root (agent_eval -> evaluation -> src -> root)
DEFAULT_JUDGE = HERE / "configs" / "judge.yaml"
KEYS = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",), "groq": ("GROQ_API_KEY",),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


# ---------------------------------------------------------------- copied from rag_eval_approach/env.py
def load_env(path=ROOT / ".env"):
    """Load KEY=value lines from the repo's .env into os.environ; variables already set win over the file."""
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


# ---------------------------------------------------------------- judge config
def read_judge(path=DEFAULT_JUDGE):
    """The judge config ({provider, model, options}) without checking for a key: enough to name the judge."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg["options"] = cfg.get("options") or {}
    return cfg


def load_judge(path=DEFAULT_JUDGE):
    """The judge config, after checking that it can be called."""
    load_env()
    cfg = read_judge(path)
    if str(cfg.get("model", "")).startswith("REPLACE"):
        raise SystemExit(f"set the judge model id in {path}")
    if not has_key(cfg["provider"]):
        raise SystemExit(f"no API key for {cfg['provider']!r}: add it to .env (see .env.example) or export it")
    return cfg


def judge_name(cfg):
    return f"{cfg['provider']}:{cfg['model']}"


# ---------------------------------------------------------------- copied from rag_eval_approach/score_ragas.py
def chat_model(provider, model, options):
    """A LangChain chat model for RAGAS."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=8000)   # no temperature: current Claude models reject it
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=options.get("temperature", 0))
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=options.get("temperature", 0))
    if provider == "ollama":                                  # added in agent_eval (see module docstring)
        from langchain_ollama import ChatOllama
        extra = {"base_url": os.environ["OLLAMA_HOST"]} if os.environ.get("OLLAMA_HOST") else {}
        # optional, from judge.yaml options: reasoning (false turns a thinking model's thinking off, which is much
        # faster) and num_predict (a cap on output tokens, so one rambling reply cannot use up the job's timeout)
        extra.update({k: options[k] for k in ("reasoning", "num_predict") if k in options})
        return ChatOllama(model=model, temperature=options.get("temperature", 0),
                          num_ctx=options.get("num_ctx", 8192), keep_alive=options.get("keep_alive", "60m"),
                          **extra)
    raise ValueError(f"no RAGAS judge for provider {provider!r}")