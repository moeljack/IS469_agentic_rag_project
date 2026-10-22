"""Steps 1 + 2: Intake and Planner.

Design: ONE structured LLM call extracts intent, entities and draft sub-queries.
Everything that needs to be *right* is then decided in plain Python:
  - deriving the Year of Assessment from a financial year end
  - deciding which fields are missing for each intent
  - wording clarifying questions (templates, not model-written)

Requires: pip install ollama pydantic     (and a pulled model: see `ollama list`)
Model: defaults to qwen3.5:4b; override without editing code:  INTAKE_MODEL=<tag> python intake_planner.py
"""
import json
import os
import re
import sys
from datetime import date
from typing import Optional
from pathlib import Path

from pydantic import ValidationError


from group_proj.IS469_agentic_rag_project.src.decomposer.zignore_decomposer_schemas import (Clarify, Entities, Intent, LLMOutput, Decompose, PlannedQuery,
                     Refuse, Result, SearchFilter)

MODEL = os.environ.get("INTAKE_MODEL", "qwen3.5:4b")
NUM_CTX = 8192

# ------------------------------------------------------------------ 1. the LLM call

# load the few_shots.json file
FEW_SHOT_PATH = Path(__file__).parent / "few_shots.json"
try:
    with open(FEW_SHOT_PATH, "r", encoding="utf-8") as f:
        raw_few_shots = json.load(f)
        # Convert JSON objects back to the (user_prompt, output_dict) tuple format expected by build_system_prompt
        FEW_SHOT = [(item["user_prompt"], item["output"]) for item in raw_few_shots]
except FileNotFoundError:
    FEW_SHOT = []  # Fallback if file isn't found


def build_system_prompt(today: date) -> str:
    shots = "\n\n".join(
        f"User: {q}\nJSON: {json.dumps(a)}" for q, a in FEW_SHOT)
    return f"""You are the intake and planning step of a Singapore corporate income tax assistant.
Today's date is {today.isoformat()}. Read the conversation and output JSON matching the schema.

Rules:
- intent: lookup (rates, rules, definitions) | eligibility (does a company qualify) |
  computation (work out tax payable or a relief) | deadline (filing / payment dates) |
  procedure (how to file or claim) | out_of_scope (personal income tax, GST, property tax,
  other countries, small talk).
- entities: include ONLY what the user explicitly stated. Use null when unknown. Never guess.
  Set "ya" only if the user says YA / Year of Assessment. Set "fy" only if they say FY / financial year.
  Dates as YYYY-MM-DD.
- topics: official scheme or form names mentioned or clearly implied.

Examples:
{shots}"""


def extract(messages: list[dict], today: date, model: str = MODEL) -> LLMOutput:
    """One structured LLM call with a single repair retry."""
    import ollama  # imported lazily so the rest of the module is testable without it

    msgs = [{"role": "system", "content": build_system_prompt(today)}, *messages]
    last_err: Optional[Exception] = None
    kwargs = dict(
        model=model,
        format=LLMOutput.model_json_schema(),   # constrained decoding to the schema
        options={"temperature": 0, "num_ctx": NUM_CTX},
    )
    for _ in range(2):
        try:
            # think=False skips the reasoning phase (needs ollama-python >= 0.5)
            resp = ollama.chat(messages=msgs, think=False, **kwargs)
        except TypeError:        # older client without the `think` argument
            resp = ollama.chat(messages=msgs, **kwargs)
        content = resp["message"]["content"]
        if os.environ.get("INTAKE_DEBUG"):
            print(f"[intake raw model output]\n{content}\n", flush=True)
        try:
            return LLMOutput.model_validate_json(content)
        except ValidationError as err:
            last_err = err
            msgs += [{"role": "assistant", "content": content},
                     {"role": "user", "content": f"That JSON was invalid: {err}. Return corrected JSON only."}]
    raise RuntimeError(f"Intake model failed schema validation twice: {last_err}")


# ------------------------------------------------------------------ 2. deterministic logic

def _normalise(e: Entities, today: date) -> Entities:
    """Validate what the model extracted and derive the YA. Bad values are dropped, not trusted."""
    e = e.model_copy(deep=True)

    if e.fye:
        try:
            e.fye = date.fromisoformat(e.fye).isoformat()
        except ValueError:
            e.fye = None
    if e.chargeable_income is not None and e.chargeable_income < 0:
        e.chargeable_income = None
    for field in ("ya", "fy", "incorporation_year"):
        v = getattr(e, field)
        if v is not None and not (2000 <= v <= today.year + 2):
            setattr(e, field, None)

    # YA = calendar year after the basis period. FYE 31 Dec 2025 (or 30 Jun 2025) -> YA 2026.
    if e.ya is None:
        if e.fy is not None:
            e.ya = e.fy + 1
        elif e.fye:
            e.ya = date.fromisoformat(e.fye).year + 1
    return e


_NEW_CO = re.compile(r"\b(newly|recently)\s+(incorporated|formed|set\s+up|established|registered)\b"
                     r"|\bnew(ly)?\s+(company|entity|start-?up)\b", re.I)
_NOT_NEW = re.compile(r"\bnot\s+(a\s+)?(newly|new|recently)\b", re.I)
_INC_YEAR = re.compile(r"\bincorporated\s+(?:in|on|since)\s+(?:[\w\s,]*?)(20\d{2})\b", re.I)
_TOPIC_PATTERNS = {
    "Start-Up Tax Exemption": re.compile(r"start-?\s?up\s+(tax\s+)?exemption|\bSUTE\b", re.I),
    "Partial Tax Exemption": re.compile(r"partial\s+(tax\s+)?exemption|\bPTE\b", re.I),
    "CIT Rebate": re.compile(r"\b(CIT|corporate\s+income\s+tax)\s+rebate\b", re.I),
    "Form C-S": re.compile(r"\bForm\s+C-S\b", re.I),
    "ECI": re.compile(r"\bECI\b|estimated\s+chargeable\s+income", re.I),
}


def text_hints(text: str) -> Entities:
    """Facts that can be read straight off the user's wording. A small model often misses these,
    so code is the backstop. Only high-precision patterns belong here."""
    h = Entities()
    if _NOT_NEW.search(text):
        h.is_new_company = False
    elif _NEW_CO.search(text):
        h.is_new_company = True
    m = _INC_YEAR.search(text)
    if m:
        h.incorporation_year = int(m.group(1))
    h.topics = [name for name, pat in _TOPIC_PATTERNS.items() if pat.search(text)]
    return h


def fill_gaps(e: Entities, hints: Entities) -> Entities:
    """Model output wins; hints only fill fields the model left empty."""
    e = e.model_copy(deep=True)
    for field in ("is_new_company", "incorporation_year"):
        if getattr(e, field) is None and getattr(hints, field) is not None:
            setattr(e, field, getattr(hints, field))
    e.topics = list(dict.fromkeys([*e.topics, *hints.topics]))
    return e


def merge_entities(old: Optional[Entities], new: Entities) -> Entities:
    """Carry facts across clarification turns. New non-empty values win."""
    if old is None:
        return new
    merged = old.model_copy(deep=True)
    for field, value in new.model_dump().items():
        if field == "topics":
            merged.topics = list(dict.fromkeys([*merged.topics, *value]))
        elif value is not None:
            setattr(merged, field, value)
    return merged


def _missing(intent: Intent, e: Entities) -> list[str]:
    """Which facts must the user supply before we can plan? Decided here, not by the model."""
    need: list[str] = []
    if intent == Intent.COMPUTATION and e.chargeable_income is None:
        need.append("chargeable_income")
    if intent == Intent.DEADLINE and e.fye is None:
        need.append("fye")
    if intent == Intent.ELIGIBILITY:
        startup = any("start" in t.lower() for t in e.topics)
        if startup and e.is_new_company is None and e.incorporation_year is None:
            need.append("incorporation")
    return need


CLARIFY_TEMPLATES = {
    "chargeable_income": "What is the company's chargeable income (in SGD) for the year you're asking about?",
    "fye": "What is the company's financial year end date (e.g. 31 December)?",
    "incorporation": "When was the company incorporated (or is it newly incorporated)?",
    "business_nature": ""
}

# TOOLS_BY_INTENT = {
#     Intent.LOOKUP: ["get_parameter"],
#     Intent.COMPUTATION: ["get_parameter", "compute_tax"],
#     Intent.DEADLINE: ["compute_deadline"],
#     Intent.ELIGIBILITY: ["check_eligibility"],
#     Intent.PROCEDURE: [],
# }


def resolve(raw: LLMOutput, question: str, today: date,
            prior: Optional[Entities] = None, max_queries: int = 4) -> Result:
    """Turn raw model output into Plan | Clarify | Refuse."""
    if raw.intent == Intent.OUT_OF_SCOPE:
        return Refuse(reason="This assistant only covers Singapore corporate income tax.")

    e = _normalise(fill_gaps(merge_entities(prior, raw.entities), text_hints(question)), today)

    missing = _missing(raw.intent, e)
    if missing:
        return Clarify(question=" ".join(CLARIFY_TEMPLATES[m] for m in missing[:2]),
                       missing=missing, entities=e)

    assumptions: list[str] = []
    ya = e.ya
    if ya is None:
        ya = today.year
        assumptions.append(f"Assumed Year of Assessment {ya} (the current YA). Tell me if you meant another year.")

    # Draft sub-queries from the model + the user's own wording as a safety net.
    # drafts = [(sq.query.strip(), sq.doc_types) for sq in raw.sub_queries if sq.query.strip()]
    # drafts.append((question.strip(), []))
    # seen, queries = set(), []
    # for text, doc_types in drafts:
    #     key = text.lower()
    #     if key in seen:
    #         continue
    #     seen.add(key)
    #     queries.append(PlannedQuery(text=text, filter=SearchFilter(ya=ya, doc_types=doc_types)))
    # queries = queries[:max_queries]

    return Decompose(intent=raw.intent, entities=e, ya=ya, assumptions=assumptions,
                # queries=queries, tools=TOOLS_BY_INTENT.get(raw.intent, [])
                )


def intake_and_plan(messages: list[dict], today: Optional[date] = None,
                    prior: Optional[Entities] = None, model: str = MODEL) -> Result:
    """Public entry point. `messages` = the conversation so far (user/assistant turns)."""
    today = today or date.today()
    question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    raw = extract(messages, today, model)
    return resolve(raw, question, today, prior)


# ------------------------------------------------------------------ quick manual try-out

if __name__ == "__main__":
    messages: list[dict] = []
    prior: Optional[Entities] = None
    while True:
        q = input("you> ").strip()
        if not q:
            break
        messages.append({"role": "user", "content": q})
        result = intake_and_plan(messages, prior=prior)
        if isinstance(result, Clarify):
            print("bot>", result.question)
            messages.append({"role": "assistant", "content": result.question})
            prior = result.entities
        else:
            print(result.model_dump_json(indent=2) if isinstance(result, Decompose) else f"bot> {result.reason}")
            messages, prior = [], None