"""The answer step every system shares (plan Section 5, "Shared settings").

The agent writes its answer with the supervisor's ANSWER_SYSTEM prompt, temperature 0, thinking off, and each chunk cut
to DOC_CHARS characters (supervisor.excerpt: the window of a long chunk that best matches the question). The naive
systems must answer the same way, or an accuracy gap would mix "agentic" with "different prompt". So this module reuses the supervisor's prompt and limits, and builds the same user message as
Supervisor._answer_messages() (same [Doc n] header, same sections) with no user facts, because the naive pipeline has
no planner to extract any. It does not change the agent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "supervisor"))

from supervisor import ANSWER_SYSTEM, DOC_CHARS, excerpt

INSUFFICIENT = "INSUFFICIENT_CONTEXT"
PROMPT_VERSION = "supervisor.ANSWER_SYSTEM"   # recorded with every run


def answer_messages(question, chunks):
    """chunks: retriever hits (dicts with text, source, headings), best first."""
    def head(i, chunk):
        where = chunk.get("headings")
        return f"[Doc {i}] (source: {chunk.get('source', '')}" + (f" | {where}" if where else "") + ")"
    body = "\n\n".join(f"{head(i, c)}\n{excerpt(c['text'], question, DOC_CHARS)}" for i, c in enumerate(chunks, 1))
    user = f"QUESTION: {question}\n\nUSER FACTS:\n- none\n" + f"\nDOCUMENTS:\n{body}"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def context_used(chunks, question):
    """The chunk text exactly as the answer prompt shows it (RAGAS scores faithfulness against this)."""
    return [{"id": c["id"], "text": excerpt(c["text"], question, DOC_CHARS)} for c in chunks]


def generate_answer(llm, question, chunks):
    """(answer text, refused). An empty reply or the INSUFFICIENT_CONTEXT marker counts as a refusal."""
    draft = llm(answer_messages(question, chunks)).strip()
    return draft, (not draft or draft.startswith(INSUFFICIENT))
