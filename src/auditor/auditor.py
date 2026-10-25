from typing import List, Literal, Optional

import ollama
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Schemas for Supervisor Routing
# ---------------------------------------------------------------------------
class ChunkEvaluation(BaseModel):
    chunk_id: str = Field(description="The exact 'id' of the chunk being evaluated (e.g., web:a5627a3f6e:1)")
    status: Literal["approved", "rejected"] = Field(
        description="'approved' if the chunk contains information that helps answer ANY PART of the question. "
                    "'rejected' only if it is off-topic, a title/metadata page, or its rules are for a different year."
    )
    # json_schema_extra goes into the schema Ollama decodes against (a length cap at generation time) without making
    # pydantic fail validation if a model overruns it: a long reason must never make the whole audit "unavailable".
    reason: str = Field(description="At most 12 words.", json_schema_extra={"maxLength": 120})

class AuditorResponse(BaseModel):
    # Order matters: the model writes fields in this order, so the verdicts come first and the overall judgement after.
    chunk_evaluations: List[ChunkEvaluation] = Field(description="Evaluation of each individual chunk.")
    is_sufficient: bool = Field(description="True only if the approved chunks together answer ALL parts of the question.")
    recommended_search_query: Optional[str] = Field(
        default=None,
        description="If is_sufficient is false: a short keyword query (max 8 words, no years) for the missing part. Otherwise null.",
    )

# ---------------------------------------------------------------------------
# Auditor Prompt
# ---------------------------------------------------------------------------
AUDITOR_PROMPT = """You are a Tax Auditor Agent for Singapore IRAS Corporate Income Tax.
Decide which retrieved chunks are useful for answering the user's question, and whether together they are enough.

### User question: {original_query}
### Context: {sub_prompt}

### Retrieved chunks:
{chunks_text}

### Rules for each chunk
- approved: the chunk contains information that helps answer ANY PART of the question (conditions, rates, caps, definitions, how a scheme works). A chunk does NOT have to answer the whole question, and it does NOT have to mention the exact year or the user's figures. A general description of a scheme counts.
- rejected: ONLY if the chunk is about a different topic (e.g. GST, insurance, advance rulings), is just a title / navigation / metadata page, or its rates or rules are explicitly for a different year than asked and would give a wrong answer.
- Never reject a chunk just because it is incomplete or lacks the user's year or numbers.
- reason: at most 12 words.

### Then
- is_sufficient: true only if the approved chunks together answer ALL parts of the question.
- recommended_search_query: if is_sufficient is false, a short keyword query (max 8 words, no years) for the missing part; otherwise null.
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
        self.keep_alive = keep_alive   # keep phi3.5 resident between turns instead of reloading it

    def _format_chunks(self, chunks: list) -> str:
        """Strips vector scores and formats the chunks for the LLM."""
        formatted = []
        for chunk in chunks:
            formatted.append(
                f"--- ID: {chunk.get('id')} ---\n"
                f"Headings: {chunk.get('headings')}\n"
                f"Text: {chunk.get('text')}\n"
            )
        return "\n".join(formatted)

    def evaluate(self, original_query: str, sub_prompt: str, chunks: list) -> AuditorResponse:
        """Called by the Supervising Agent to evaluate a specific sub-prompt."""

        if not chunks:
            return AuditorResponse(
                is_sufficient=False,
                chunk_evaluations=[],
                recommended_search_query=sub_prompt
            )

        prompt = AUDITOR_PROMPT.format(
            original_query=original_query,
            sub_prompt=sub_prompt,
            chunks_text=self._format_chunks(chunks)
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

# ---------------------------------------------------------------------------
# Execution Example
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     auditor = AuditorAgent(model="phi3.5:3.8b")    # talks to the local Ollama server

#     # Example Input from your Retriever
#     sample_chunks = [
#         {
#             "id": "web:a5627a3f6e:1",
#             "text": "You can use your credit card to pay your tax on AXS e-Station... payment limit of $20,000.00.",
#             "headings": "Payments > Credit Card Payment"
#         }
#     ]

#     result = auditor.evaluate(
#         original_query="Can I pay via credit card and what happens if I am late?",
#         sub_prompt="Can corporate tax be paid by credit card?",
#         chunks=sample_chunks
#     )

#     print(result.model_dump_json(indent=2))