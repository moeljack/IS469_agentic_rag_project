"""Gemini adapter: llm(messages, schema=None) -> str on the google-genai SDK (the plan's suggested third-party judge).

With a schema the reply is requested as JSON (response_mime_type); the prompt itself spells out the JSON shape.
Needs `pip install google-genai` and GEMINI_API_KEY (or GOOGLE_API_KEY).
"""
from google import genai
from google.genai import types

from src.evaluation.rag_eval.rag_eval_approach.llm import split_system
from src.evaluation.rag_eval.rag_eval_approach.tracking import USAGE


def google_llm(model, temperature=0):
    client = genai.Client()

    def llm(messages, schema=None):
        system, rest = split_system(messages)
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
                    for m in rest]
        config = types.GenerateContentConfig(system_instruction=system or None, temperature=temperature,
                                             response_mime_type="application/json" if schema else None)
        response = client.models.generate_content(model=model, contents=contents, config=config)
        meta = response.usage_metadata
        USAGE.add(getattr(meta, "prompt_token_count", 0), getattr(meta, "candidates_token_count", 0))
        return response.text or ""

    return llm
