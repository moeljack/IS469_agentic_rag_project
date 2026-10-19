"""Answer a question from chunks retrieved out of ChromaDB.

Flow: question -> Retriever.search_tiered (web pages first, PDFs as fallback) -> numbered context -> Ollama -> answer
with [Doc n] citations, plus the list of sources behind those numbers.

Run:  python src/inference/generate.py "What is the corporate income tax rate?"
      python src/inference/generate.py            (interactive)
Model: defaults to qwen3.5:4b; override with  RAG_MODEL=<tag>
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

MODEL = os.environ.get("RAG_MODEL", "qwen3.5:4b")
NUM_CTX = 8192
MAX_CONTEXT_CHARS = 14000        # ~3.5k tokens of context, leaves room for prompt + answer in an 8k window
TOP_K = 5
MIN_SCORE = 0.5                  # reranker 0-1 relevance; weaker chunks are not shown to the model
INSUFFICIENT = "INSUFFICIENT_CONTEXT: The provided documents do not contain this information."

SYSTEM_PROMPT = f"""You are an assistant for Singapore corporate income tax, answering strictly from the excerpts provided.
Rules:
1. Use ONLY facts explicitly stated inside the <context> tags. Do not use outside knowledge or guess.
2. Cite every claim with the document tag it came from, e.g. [Doc 1] or [Doc 2][Doc 3].
3. If the context does not contain enough information to answer, reply exactly with:
   {INSUFFICIENT}
4. Quote figures, dates and conditions exactly as written (rates, deadlines, caps, Year of Assessment).
5. If the excerpts conflict, prefer the one that is more specific to the question and mention the conflict.
6. Be concise: a direct answer first, then supporting details if needed."""


@dataclass
class Answer:
    question: str
    answer: str
    hits: list = field(default_factory=list)       # retrieved chunks shown to the model, in [Doc n] order
    from_fallback: bool = False                    # True when the web pages did not answer and PDFs were used

    def sources(self) -> list[str]:
        return [f"[Doc {i}] {_source_label(h)}" for i, h in enumerate(self.hits, 1)]


# ---------------------------------------------------------------- retrieval

def retrieve(retriever, queries: list[str], k: int = TOP_K, min_score: Optional[float] = MIN_SCORE) -> list[dict]:
    """Run one or more queries and merge the hits (best score per chunk), keeping the top k.

    Several queries come from the intake planner's sub-queries; a single question works too.
    """
    merged: dict[str, dict] = {}
    for q in queries:
        for h in retriever.search_tiered(q, k=k, min_score=min_score):
            if h["id"] not in merged or h["score"] > merged[h["id"]]["score"]:
                merged[h["id"]] = h
    hits = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
    # never mix tiers: if any web hit survived, drop PDF fallbacks from other queries
    if any(not h["fallback"] for h in hits):
        hits = [h for h in hits if not h["fallback"]]
    return hits[:k]


# ---------------------------------------------------------------- prompt building

def _source_label(h: dict) -> str:
    if h["source_type"] == "webpage":
        return f"{h.get('title') or 'IRAS page'} - {h['source']}"
    pages = f", pages {h['pages']}" if h.get("pages") else ""
    return f"{h['source']}{pages}"


def format_context(hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
    """Numbered passages, truncated to a character budget. Returns (context, hits actually included)."""
    blocks, used, total = [], [], 0
    for h in hits:
        header = f"[Doc {len(used) + 1}] (Source: {_source_label(h)})"
        if h.get("headings"):
            header += f"\nSection: {h['headings']}"
        block = f"{header}\n{h['text'].strip()}"
        if used and total + len(block) > max_chars:
            break                                   # always keep at least the best chunk
        blocks.append(block[:max_chars])
        used.append(h)
        total += len(block)
    return "\n\n".join(blocks), used


def build_messages(question: str, context: str, history: Optional[list[dict]] = None) -> list[dict]:
    user = f"<context>\n{context}\n</context>\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": user}]


# ---------------------------------------------------------------- generation

def chat(messages: list[dict], model: str = MODEL, temperature: float = 0.1) -> str:
    import ollama   # lazy: the rest of the module stays importable without it
    kwargs = dict(model=model, messages=messages, options={"temperature": temperature, "num_ctx": NUM_CTX})
    try:
        resp = ollama.chat(think=False, **kwargs)   # skip the reasoning phase (ollama-python >= 0.5)
    except TypeError:
        resp = ollama.chat(**kwargs)
    msg = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
    return (msg or "").strip()


def answer_question(retriever, question: str, queries: Optional[list[str]] = None, model: str = MODEL,
                    k: int = TOP_K, min_score: Optional[float] = MIN_SCORE,
                    history: Optional[list[dict]] = None) -> Answer:
    """Retrieve, build the prompt, generate.

    queries: retrieval queries (e.g. Plan.queries texts). Defaults to the question itself.
    """
    hits = retrieve(retriever, queries or [question], k=k, min_score=min_score)
    if not hits:   # nothing relevant enough: do not let the model improvise
        return Answer(question, INSUFFICIENT, [])

    context, used = format_context(hits)
    text = chat(build_messages(question, context, history), model=model)
    return Answer(question, text, used, from_fallback=any(h["fallback"] for h in used))


def print_answer(a: Answer) -> None:
    print(f"\nQ: {a.question}")
    if a.from_fallback:
        print("(not found on the web pages - answered from the PDFs)")
    print(f"\n{a.answer}\n")
    if a.hits and not a.answer.startswith("INSUFFICIENT_CONTEXT"):
        print("Sources:")
        for line in a.sources():
            print(f"  {line}")


if __name__ == "__main__":
    from src.retrieval.retriever import Retriever

    retriever = Retriever()   # loads embedder (+ reranker on first query) once
    if len(sys.argv) > 1:
        print_answer(answer_question(retriever, " ".join(sys.argv[1:])))
    else:
        while (q := input("you> ").strip()):
            print_answer(answer_question(retriever, q))