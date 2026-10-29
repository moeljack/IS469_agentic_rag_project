"""Per-question accounting of LLM calls and tokens.

The agent's parts call Ollama in three different ways (the supervisor and decomposer through `ollama.chat`, the auditor
through `ollama.Client().chat`), and SupervisorResult does not report tokens. track_ollama() wraps both entry points
once, so every call adds its prompt and output token counts to USAGE. The API adapters (llm_*.py) add theirs directly.
Call USAGE.reset() before a question and USAGE.snapshot() after it.
"""
import functools


class Usage:
    def __init__(self):
        self.reset()

    def reset(self):
        self.input = self.output = self.calls = 0

    def add(self, input_tokens, output_tokens):
        self.input += input_tokens or 0
        self.output += output_tokens or 0
        self.calls += 1

    def snapshot(self):
        return {"input": self.input, "output": self.output, "calls": self.calls}


USAGE = Usage()


def _counted(chat):
    @functools.wraps(chat)
    def wrapper(*args, **kwargs):
        resp = chat(*args, **kwargs)
        USAGE.add(getattr(resp, "prompt_eval_count", 0), getattr(resp, "eval_count", 0))
        return resp
    wrapper._eval_counted = True
    return wrapper


def track_ollama():
    """Count tokens for every Ollama chat call in this process. Safe to call more than once."""
    import ollama
    if not getattr(ollama.chat, "_eval_counted", False):
        ollama.chat = _counted(ollama.chat)                 # module-level helper (supervisor, decomposer)
    if not getattr(ollama.Client.chat, "_eval_counted", False):
        ollama.Client.chat = _counted(ollama.Client.chat)   # client method (auditor)
