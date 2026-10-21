import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

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
    def __init__(self, client, model: str = "phi3.5"):
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
                chunk_evaluations=[],
                recommended_search_query=sub_prompt
            )

        prompt = AUDITOR_PROMPT.format(
            original_query=original_query,
            sub_prompt=sub_prompt,
            chunks_text=self._format_chunks(chunks)
        )

        # Using Ollama's local OpenAI-compatible endpoint
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AuditorResponse,
            temperature=0.0, # Temperature 0 is critical for strict logic
        )

        return response.choices[0].message.parsed

# ---------------------------------------------------------------------------
# Execution Example
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Point the OpenAI client to your local Ollama instance
#     local_client = OpenAI(
#         base_url="http://localhost:11434/v1",
#         api_key="ollama", # Required by the SDK, but ignored by Ollama
#     )

#     auditor = AuditorAgent(client=local_client, model="phi3.5")

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