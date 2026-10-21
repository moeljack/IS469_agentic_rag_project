import json
from typing import List, Literal, Optional
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
AUDITOR_PROMPT = """You are a Tax Auditor Agent for IRAS Corporate Income Tax.
Evaluate if the retrieved chunks sufficiently answer the current sub-prompt.

### Overarching User Query: {original_query}
### Current Sub-Prompt: {sub_prompt}

### Retrieved Chunks:
{chunks_text}

### Instructions:
1. Review EACH chunk individually.
2. Determine if the chunk is useful for answering the current sub-prompt (keep the overarching query in mind for context, like specific YA or entity types).
3. Assign a status of "approved" or "rejected" to each chunk_id.
4. Provide a precise reason for your decision.
"""

# ---------------------------------------------------------------------------
# Auditor Agent Implementation
# ---------------------------------------------------------------------------
class AuditorAgent:
    def __init__(self, client, model: str = "gpt-4o-mini"):
        """client: standard OpenAI client supporting beta.chat.completions.parse"""
        self.client = client
        self.model = model

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
        
        # FIXED: Matches the updated AuditorResponse schema
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

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AuditorResponse,
            temperature=0.0,
        )

        return response.choices[0].message.parsed