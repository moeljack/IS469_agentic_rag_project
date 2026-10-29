"""OpenAI adapter: llm(messages, schema=None) -> str on the chat completions API.

A schema becomes `response_format` type json_schema. `temperature` defaults to 0; set it to null in the config for
models that reject it. Needs `pip install openai` and OPENAI_API_KEY.

Also serves Groq (provider "groq" in llm.py): Groq's API is OpenAI-compatible, so the same code runs with
base_url https://api.groq.com/openai/v1 and GROQ_API_KEY. Groq hosts OpenAI's open-weight gpt-oss models, not GPT-4/5.
"""
import os

from openai import OpenAI

from src.evaluation.rag_eval.rag_eval_approach.tracking import USAGE


def openai_llm(model, temperature=0, max_tokens=None, base_url=None, api_key_env=None):
    client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env] if api_key_env else None)

    def llm(messages, schema=None):
        kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        USAGE.add(usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        return response.choices[0].message.content or ""

    return llm
