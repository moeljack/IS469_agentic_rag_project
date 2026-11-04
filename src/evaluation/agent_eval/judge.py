"""The fixed LLM judge for agent_eval's LLM-graded scores (task success: faithfulness and answer relevance).

json_judge(cfg) returns call(prompt, schema) -> reply text. For Ollama (the cluster's local judge) it calls the server
directly with the JSON schema as `format` and thinking off; for an API judge it uses a LangChain chat model.

Copied from rag_eval so agent_eval does not import it:
    load_env, has_key, KEYS   <- rag_eval/rag_eval_approach/env.py
    chat_model                <- rag_eval/rag_eval_approach/score_ragas.py (API providers only)

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
    """A LangChain chat model for an API judge (needs the matching langchain-* package; Ollama does not use this)."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=8000)   # no temperature: current Claude models reject it
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=options.get("temperature", 0))
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=options.get("temperature", 0))
    raise ValueError(f"no judge for provider {provider!r}; choose ollama, anthropic, openai or google")


# ---------------------------------------------------------------- the judge call
def json_judge(cfg, timeout=900):
    """call(prompt, schema) -> raw reply text, for judge prompts that must come back as JSON.

    Ollama: the schema is sent as `format`, so the server constrains decoding to valid JSON of that shape, and thinking
    is off (as rag_eval's supervisor.default_llm does for its judge). This is what makes a 12B local judge reliable.
    judge.yaml options used: temperature (0), num_ctx (8192), num_predict (cap on output tokens), reasoning (false),
    keep_alive (60m). Other providers: the LangChain chat model, with the reply parsed afterwards."""
    provider, model, options = cfg["provider"], cfg["model"], cfg.get("options") or {}
    if provider == "ollama":
        import ollama
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST") or None, timeout=timeout)
        opts = {"temperature": options.get("temperature", 0), "num_ctx": options.get("num_ctx", 8192)}
        if "num_predict" in options:
            opts["num_predict"] = options["num_predict"]
        think = options.get("reasoning", False)

        def call(prompt, schema):
            kw = dict(model=model, messages=[{"role": "user", "content": prompt}], format=schema, options=opts,
                      keep_alive=options.get("keep_alive", "60m"))
            try:
                resp = client.chat(think=think, **kw)
            except TypeError:                                  # older ollama client without `think`
                resp = client.chat(**kw)
            except ollama.ResponseError as e:                  # a model that rejects the thinking flag
                if "think" not in str(e).lower():
                    raise
                resp = client.chat(**kw)
            return resp["message"]["content"]
        return call

    llm = chat_model(provider, model, options)

    def call(prompt, schema):
        return llm.invoke(prompt).content
    return call