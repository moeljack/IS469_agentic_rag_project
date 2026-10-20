import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Schemas for Supervisor Routing
# ---------------------------------------------------------------------------
class RequirementCheck(BaseModel):
    requirement: str = Field(description="A distinct factual aspect of the sub-prompt.")
    is_addressed: bool = Field(description="True if the chunks explicitly contain the answer.")
    evidence_or_gap: str = Field(description="Quote from text, or description of missing data.")

class AuditorResponse(BaseModel):
    is_sufficient: bool = Field(
        description="True if the sub-prompt is fully answered without hallucination. False otherwise."
    )
    analysis: List[RequirementCheck] = Field(
        description="Step-by-step check of the sub-prompt's requirements."
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Specific tax terms, dates, or rules missing from the text."
    )
    recommended_search_query: Optional[str] = Field(
        default=None,
        description="If insufficient, an optimized keyword query for the Retrieval Agent."
    )

# ---------------------------------------------------------------------------
# Auditor Prompt
# ---------------------------------------------------------------------------
AUDITOR_PROMPT = """You are the Tax Auditor Agent in an IRAS Corporate Income Tax RAG system. 
The Supervising Agent has broken down a complex user query into smaller sub-prompts. 
Your job is to evaluate if the retrieved chunks sufficiently answer the CURRENT SUB-PROMPT.

### CONTEXT
Original Overarching Query: {original_query}
Current Sub-Prompt to Evaluate: {sub_prompt}

### RETRIEVED CHUNKS
{chunks_text}

### EVALUATION RULES
1. Evaluate ONLY against the Current Sub-Prompt, but use the Original Query for context (e.g., if the original query mentions "YA 2026", ensure the rules retrieved apply to YA 2026).
2. Tax constraints matter: If the sub-prompt asks for a deadline, percentage, or specific form, and the chunks only provide general summaries, mark `is_sufficient` as False.
3. If `is_sufficient` is False, provide a `recommended_search_query` that the Retrieval Agent can use to find the missing data.
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
        
        if not chunks:
            return AuditorResponse(
                is_sufficient=False,
                analysis=[],
                missing_information=["No context retrieved."],
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