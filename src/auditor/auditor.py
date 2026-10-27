from typing import List, Literal, Optional, Sequence

import ollama
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic Schemas for Supervisor Routing
# ---------------------------------------------------------------------------
# Every field the model writes costs generation time, which is most of an audit's latency. So chunks are referred
# to by their number in the prompt (1, 2, 3) instead of copying ids like web:a5627a3f6e:1, and only rejections carry
# a reason. json_schema_extra caps lengths in the schema Ollama decodes against without making pydantic fail
# validation if a model overruns them: an over-long field must never make the whole audit "unavailable".
def _cap(n: int):
    """json_schema_extra for an Optional[str] field: put maxLength on the string branch of its anyOf, where a
    constrained decoder applies it (on the outer anyOf it can be ignored)."""
    def extra(schema: dict) -> None:
        for branch in schema.get("anyOf", [schema]):
            if branch.get("type") == "string":
                branch["maxLength"] = n
    return extra


_EMPTY = {"", "null", "none", "n/a", "na", "-"}


def _none_if_empty(v):
    """Small models often write the word "null" (or "none") as a string instead of a JSON null: treat it as null."""
    if isinstance(v, str) and v.strip().strip('"').lower() in _EMPTY:
        return None
    return v


class ChunkEvaluation(BaseModel):
    chunk: int = Field(description="The chunk's number, as shown in its '--- Chunk n ---' header.")
    status: Literal["approved", "rejected"] = Field(
        description="'approved' if the chunk contains information that helps answer ANY PART of the question. "
                    "'rejected' only if it is off-topic, a title/metadata page, or ALL of its rules are for other years."
    )
    reason: Optional[str] = Field(default=None, description="Only for rejected chunks: why, in at most 8 words. "
                                                            "null for approved chunks.",
                                  json_schema_extra=_cap(60))

    _clean_reason = field_validator("reason", mode="before")(_none_if_empty)


class AuditorResponse(BaseModel):
    # Order matters: the model writes fields in this order, so the verdicts come first and the overall judgement after.
    chunk_evaluations: List[ChunkEvaluation] = Field(description="Evaluation of each individual chunk.")
    is_sufficient: bool = Field(description="True only if the approved chunks together answer ALL parts of the question.")
    missing: Optional[str] = Field(
        default=None,
        description="If is_sufficient is false: the fact still missing, in at most 8 words. Otherwise null.",
        json_schema_extra=_cap(80),
    )
    recommended_search_query: Optional[str] = Field(
        default=None,
        description="If is_sufficient is false: a short keyword query (max 8 words, no years) for the missing fact, "
                    "different from every query already searched. Otherwise null.",
        json_schema_extra=_cap(80),
    )

    _clean = field_validator("missing", "recommended_search_query", mode="before")(_none_if_empty)

# ---------------------------------------------------------------------------
# Auditor Prompt
# ---------------------------------------------------------------------------
AUDITOR_PROMPT = """You are a Tax Auditor Agent for Singapore IRAS Corporate Income Tax.
Decide which retrieved chunks are useful for answering the user's question, and whether together they are enough.

### User question: {original_query}
### Year of Assessment asked: {year}
### Context: {sub_prompt}

### Queries already searched (never suggest these again, or rephrasings of them):
{tried_queries}

### Retrieved chunks:
{chunks_text}

### Rules for each chunk (refer to each chunk by its number)
- approved: the chunk contains information that helps answer ANY PART of the question (conditions, rates, caps, definitions, how a scheme works). A chunk does NOT have to answer the whole question, and it does NOT have to mention the exact year or the user's figures. A general description of a scheme counts.
- rejected: ONLY if the chunk is about a different topic (e.g. GST, insurance, advance rulings), is just a title / navigation / metadata page, or EVERYTHING in it applies only to years other than the year asked.
- Mixed years: a chunk that shows the current rules next to older ones (for example a "YA 2020 onwards" table and a "YA 2019 and before" table) is APPROVED, because its current part is useful. A heading that names an older year does not make the whole chunk outdated.
- Never reject a chunk just because it is incomplete or lacks the user's year or numbers.
- reason: only for rejected chunks, at most 8 words. Use null for approved chunks.

### Then
- is_sufficient: true only if the approved chunks together answer ALL parts of the question.
- missing: if is_sufficient is false, name the fact that is still missing in at most 8 words (e.g. "shareholder count limit"); otherwise null.
- recommended_search_query: if is_sufficient is false, a short keyword query (max 8 words, no years) that would find the missing fact and is NOT one of the queries already searched; otherwise null.
Return JSON matching the schema.
"""

# ---------------------------------------------------------------------------
# Auditor Agent Implementation
# ---------------------------------------------------------------------------
class AuditorAgent:
    def __init__(self, client=None, model: str = "phi3.5:3.8b", num_ctx: int = 8192, keep_alive: str = "30m",
                 think=None):
        """client: an ollama.Client (defaults to the local Ollama server)."""
        self.client = client or ollama.Client()
        self.model = model
        self.num_ctx = num_ctx
        self.think = think             # False for thinking models like qwen3.5; None = don't send the flag
        self.keep_alive = keep_alive   # keep the model resident between turns instead of reloading it

    @staticmethod
    def _format_chunks(chunks: list) -> str:
        """Numbers the chunks 1..n for the model (ids and scores are left out: the model never needs to copy them)."""
        return "\n".join(f"--- Chunk {n} ---\nHeadings: {chunk.get('headings')}\nText: {chunk.get('text')}\n"
                         for n, chunk in enumerate(chunks, 1))

    @staticmethod
    def _format_tried(tried_queries: Sequence[str]) -> str:
        return "\n".join(f"- {q}" for q in tried_queries) or "- none yet"

    def evaluate(self, original_query: str, sub_prompt: str, chunks: list, tried_queries: Sequence[str] = (),
                 year: Optional[int] = None) -> AuditorResponse:
        """Called by the Supervising Agent. In the result, chunk_evaluations[*].chunk is the 1-based position of the
        chunk in `chunks` (use chunk_id_of() to map it back to an id).

        tried_queries: queries already searched for this question, so the suggestion is something new.
        year:          the Year of Assessment asked about, so the auditor can tell current rules from older ones.
        """
        if not chunks:
            return AuditorResponse(
                is_sufficient=False,
                chunk_evaluations=[],
                recommended_search_query=sub_prompt
            )

        prompt = AUDITOR_PROMPT.format(
            original_query=original_query,
            year=f"YA {year}" if year else "not stated",
            sub_prompt=sub_prompt,
            tried_queries=self._format_tried(tried_queries),
            chunks_text=self._format_chunks(chunks),
        )

        # Ollama's native client: the schema constrains decoding, so the reply is always parseable JSON.
        # (`think` is sent only if set: phi3.5 is not a reasoning model and Ollama may reject the flag; qwen3.5 needs think=False.)
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=AuditorResponse.model_json_schema(),
            options={"temperature": 0.0, "num_ctx": self.num_ctx},  # temperature 0 is critical for strict logic
            keep_alive=self.keep_alive,
        )
        if self.think is not None:
            kwargs["think"] = self.think
        response = self.client.chat(**kwargs)
        return AuditorResponse.model_validate_json(response["message"]["content"])

    @staticmethod
    def chunk_id_of(evaluation: ChunkEvaluation, chunks: list) -> Optional[str]:
        """The id of the chunk an evaluation refers to, or None if the model gave a number outside 1..len(chunks)."""
        n = evaluation.chunk
        return chunks[n - 1].get("id") if 1 <= n <= len(chunks) else None

# ---------------------------------------------------------------------------
# Execution Example
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     auditor = AuditorAgent(model="qwen3.5:4b", think=False)    # talks to the local Ollama server
#
#     # Example Input from your Retriever
#     sample_chunks = [
#         {
#             "id": "web:a5627a3f6e:1",
#             "text": "You can use your credit card to pay your tax on AXS e-Station... payment limit of $20,000.00.",
#             "headings": "Payments > Credit Card Payment"
#         }
#     ]
#
#     result = auditor.evaluate(
#         original_query="Can I pay via credit card and what happens if I am late?",
#         sub_prompt="Can corporate tax be paid by credit card?",
#         chunks=sample_chunks,
#         tried_queries=["corporate tax credit card payment"],
#         year=2026,
#     )
#
#     print(result.model_dump_json(indent=2))