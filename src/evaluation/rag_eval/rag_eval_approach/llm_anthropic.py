"""Claude adapter: llm(messages, schema=None) -> str on the Anthropic Messages API.

- The system prompt goes in the separate `system` field.
- A schema becomes structured output (`output_config.format`, type json_schema), so the reply is valid JSON text.
  Structured outputs want `additionalProperties: false` on every object, so it is added to a copy of the schema.
- No temperature: current Claude models reject sampling parameters. Thinking stays at the model's default (adaptive);
  `effort` (low/medium/high/xhigh/max) is the only depth control, set from the config if given.
- A refusal (stop_reason "refusal") returns "", which the supervisor treats as unusable output and the scorers count
  as an empty answer.
- No server-side refusal fallback: it would let a different model answer some questions and blur the comparison.

Needs `pip install anthropic` and ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
import copy

import anthropic

from src.evaluation.rag_eval.rag_eval_approach.llm import split_system
from src.evaluation.rag_eval.rag_eval_approach.tracking import USAGE


def _strict(schema):
    schema = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(schema)
    return schema


def anthropic_llm(model, effort=None, max_tokens=16000):
    client = anthropic.Anthropic()

    def llm(messages, schema=None):
        system, rest = split_system(messages)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": rest}
        if system:
            kwargs["system"] = system
        output_config = {}
        if effort:
            output_config["effort"] = effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": _strict(schema)}
        if output_config:
            kwargs["output_config"] = output_config
        response = client.messages.create(**kwargs)
        USAGE.add(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")

    return llm
