from typing import List, Literal, Optional

import ollama
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Schemas for Supervisor Routing
# ---------------------------------------------------------------------------
class ChunkEvaluation(BaseModel):
    chunk_id: str = Field(description="The exact 'id' of the chunk being evaluated (e.g., web:a5627a3f6e:1)")
    status: Literal["approved", "rejected"] = Field(
        description="'approved' if it contains relevant facts to answer the prompt, 'rejected' if irrelevant or insufficient."
    )
    reason: str = Field(
        description="Brief explanation of why it was approved or rejected (e.g., 'Missing late penalty information')."
    )

class AuditorResponse(BaseModel):
    is_sufficient: bool = Field(description="True if the approved chunks combined fully answer the prompt.")
    chunk_evaluations: List[ChunkEvaluation] = Field(description="Evaluation of each individual chunk.")
    recommended_search_query: Optional[str] = Field(default=None)

# ---------------------------------------------------------------------------
# Auditor Prompt
# ---------------------------------------------------------------------------
# Notice we add "return as JSON" to help the smaller model lock into the format
AUDITOR_PROMPT = """You are a Tax Auditor Agent for IRAS Corporate Income Tax.
Evaluate if the retrieved chunks sufficiently answer the current sub-prompt.

### Overarching User Query: {original_query}
### Current Sub-Prompt: {sub_prompt}

### Retrieved Chunks:
{chunks_text}

### Instructions:
1. Review EACH chunk individually.
2. Determine if the chunk is useful for answering the current sub-prompt.
3. Assign a status of "approved" or "rejected" to each chunk_id.
4. Provide a precise reason for your decision.
5. Return your response strictly as JSON matching the requested schema.
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