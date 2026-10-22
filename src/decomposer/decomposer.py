import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from pydantic import ValidationError

from decomposer_general_schemas import (
    TaxDomain,
    Intent,
    CITEntities,
    PropertyTaxEntities,
    GSTEntities,
    StampDutyEntities,
    IndividualTaxEntities,
    DomainEntities,
    LLMOutput,
    Clarify,
    Refuse,
    Decompose,
    PlannedQuery,
    SearchFilter,
    Result,
)

MODEL = os.environ.get("INTAKE_MODEL", "qwen3.5:4b")
NUM_CTX = 8192

FEW_SHOT_PATH = Path(__file__).parent / "few_shots.txt"

# ------------------------------------------------------------------ 1. System Prompt & Extraction
def load_few_shots_from_txt(filepath: Path | str = FEW_SHOT_PATH) -> list[dict]:
    """Parses few_shots.txt into role-based messages for Ollama."""
    path = Path(filepath)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    blocks = content.split("=== EXAMPLE ")
    messages: list[dict] = []

    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        current_role = None
        current_content: list[str] = []

        for line in lines:
            if line.startswith("==="):
                continue
            if line.strip() == "<|user|>":
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip(),
                    })
                    current_content = []
                current_role = "user"
            elif line.strip() == "<|assistant|>":
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": "\n".join(current_content).strip(),
                    })
                    current_content = []
                current_role = "assistant"
            else:
                current_content.append(line)

        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_content).strip(),
            })

    return messages

FEW_SHOT_MESSAGES = load_few_shots_from_txt(FEW_SHOT_PATH)

def build_system_prompt(today: date) -> str:
    return f"""You are the intake and planning stage of an official Singapore IRAS Tax assistant.
Today's date is {today.isoformat()}.
Ensure dates entered by users are actual dates. Dates like 33 October should not be accepted.
Analyze the conversation and output valid JSON strictly matching the schema.

Domains:
- cit (Corporate Income Tax: Form C/C-S, SUTE, PTE, CIT rebate, ECI)
- property_tax (Annual Value, Owner-Occupier rates, Non-owner residential, Commercial)
- gst (Registration $1M threshold, standard-rated 9%, reverse charge)
- stamp_duty (BSD, ABSD, SSD, lease duties)
- individual_tax (Personal income tax, tax residency, reliefs, SRS)
- general (General IRAS administrative enquiries)
- out_of_scope (Foreign taxes, non-tax queries)

Intents: lookup | eligibility | computation | deadline | procedure | out_of_scope

Rules:
1. CONVERSATION CONTINUITY (CRITICAL):
   - Review the entire dialog turn-by-turn. Do NOT treat the last message in isolation.
   - If the previous assistant turn asked for a specific figure and the user replies with a number, currency, or short answer (e.g., "310k", "$350,000", "2m"), map that value DIRECTLY to the requested field.
   - Maintain the overarching intent (e.g., computation) across follow-up turns unless the user explicitly changes topics.

2. NUMERIC & CURRENCY PARSING:
   - Normalize numeric abbreviations and commas into clean numbers:
     * "k" / "thousand" = x 1,000 (e.g., "310k" -> 310000.0, "$45k" -> 45000.0)
     * "m" / "mil" / "million" = x 1,000,000 (e.g., "1.2m" -> 1200000.0)
     * Strip commas (e.g., "350,000" -> 350000.0)
   - For computation requests: If the user states "we earned $X", "our profit is $X", or "income of $X", extract this value as "chargeable_income".

3. SCHEMA RESTRICTIONS:
   - Populate ONLY the entity sub-object corresponding to the selected domain. Set all other domain fields to null.
   - Extract ONLY explicit facts mentioned by the user across the conversation. Never guess unmentioned fields.
   - Generate 1-3 targeted sub-queries aimed at IRAS regulations, rates, or circulars.

"""


def extract(messages: list[dict], today: date, model: str = MODEL) -> LLMOutput:
    import ollama

    msgs = [
        {"role": "system", "content": build_system_prompt(today)},
        *FEW_SHOT_MESSAGES,
        *messages,
    ]
    last_err: Optional[Exception] = None
    kwargs = dict(
        model=model,
        format=LLMOutput.model_json_schema(),
        options={"temperature": 0, "num_ctx": NUM_CTX},
    )

    for _ in range(2):
        try:
            resp = ollama.chat(messages=msgs, think=False, **kwargs)
        except TypeError:
            resp = ollama.chat(messages=msgs, **kwargs)

        content = resp["message"]["content"]
        if os.environ.get("INTAKE_DEBUG"):
            print(f"[intake raw model output]\n{content}\n", flush=True)

        try:
            return LLMOutput.model_validate_json(content)
        except ValidationError as err:
            last_err = err
            msgs += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Schema validation failed: {err}. Return corrected JSON only."}
            ]
    raise RuntimeError(f"Intake model failed schema validation twice: {last_err}")


# ------------------------------------------------------------------ 2. Deterministic Verification & Requirements

MANDATORY_SLOTS: Dict[Tuple[TaxDomain, Intent], List[str]] = {
    # CIT
    (TaxDomain.CIT, Intent.COMPUTATION): ["chargeable_income"],
    (TaxDomain.CIT, Intent.DEADLINE): ["fye"],
    # Property Tax
    (TaxDomain.PROPERTY_TAX, Intent.COMPUTATION): ["annual_value", "is_owner_occupied"],
    # GST
    (TaxDomain.GST, Intent.COMPUTATION): ["taxable_turnover"],
    # Stamp Duty
    (TaxDomain.STAMP_DUTY, Intent.COMPUTATION): ["purchase_price_or_market_value", "buyer_profile"],
}


CLARIFY_TEMPLATES: Dict[str, str] = {
    "chargeable_income": "What is the company's estimated or actual chargeable income (in SGD)?",
    "fye": "What is the company's financial year end date (e.g., 31 December)?",
    "incorporation_year": "When was the company incorporated, and is it in its first 3 consecutive YAs?",
    "annual_value": "What is the property's Annual Value (AV) as shown on your IRAS Valuation Notice?",
    "is_owner_occupied": "Is this residential property owner-occupied (you live in it) or rented out / vacant?",
    "taxable_turnover": "What is your total or projected taxable turnover over a 12-month period?",
    "purchase_price_or_market_value": "What is the purchase price or market valuation of the property?",
    "buyer_profile": "What is the buyer's status (Singapore Citizen, PR, Foreigner, or Entity)?",
}


def _normalise_cit(e: CITEntities, today: date) -> Tuple[CITEntities, Optional[int], list[str]]:
    e = e.model_copy(deep=True)
    assumptions = []

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

    # In Singapore CIT, preceding calendar basis applies (FYE in 2025 -> YA 2026)
    derived_ya = e.ya
    if derived_ya is None:
        if e.fy is not None:
            derived_ya = e.fy + 1
        elif e.fye:
            derived_ya = date.fromisoformat(e.fye).year + 1
        else:
            derived_ya = today.year
            assumptions.append(f"Defaulted to Year of Assessment (YA) {derived_ya}.")
    e.ya = derived_ya
    return e, derived_ya, assumptions


def _normalise_property_tax(e: PropertyTaxEntities, today: date) -> Tuple[PropertyTaxEntities, int, list[str]]:
    e = e.model_copy(deep=True)
    assumptions = []

    if e.annual_value is not None and e.annual_value < 0:
        e.annual_value = None

    tax_year = e.tax_year or today.year
    if e.tax_year is None:
        assumptions.append(f"Assumed property tax assessment year {tax_year}.")
    e.tax_year = tax_year
    return e, tax_year, assumptions


def normalise_entities(domain: TaxDomain, raw_entities: DomainEntities, today: date) -> Tuple[DomainEntities, Optional[int], list[str]]:
    if domain == TaxDomain.CIT and isinstance(raw_entities, CITEntities):
        return _normalise_cit(raw_entities, today)
    if domain == TaxDomain.PROPERTY_TAX and isinstance(raw_entities, PropertyTaxEntities):
        return _normalise_property_tax(raw_entities, today)
    return raw_entities, today.year, []


def merge_domain_entities(old: Optional[DomainEntities], new: DomainEntities) -> DomainEntities:
    if old is None or type(old) is not type(new):
        return new
    merged = old.model_copy(deep=True)
    for field, value in new.model_dump().items():
        if field == "topics":
            merged.topics = list(dict.fromkeys([*getattr(merged, "topics", []), *value]))
        elif value is not None:
            setattr(merged, field, value)
    return merged


def get_active_entities(raw: LLMOutput) -> DomainEntities:
    mapping = {
        TaxDomain.CIT: (raw.cit, CITEntities),
        TaxDomain.PROPERTY_TAX: (raw.property_tax, PropertyTaxEntities),
        TaxDomain.GST: (raw.gst, GSTEntities),
        TaxDomain.STAMP_DUTY: (raw.stamp_duty, StampDutyEntities),
        TaxDomain.INDIVIDUAL_TAX: (raw.individual_tax, IndividualTaxEntities),
    }
    obj, fallback_cls = mapping.get(raw.domain, (None, CITEntities))
    return obj if obj is not None else fallback_cls()


# ------------------------------------------------------------------ 3. Planner & Entry Point

    """resolve will never split the user's prompt into more than 4 queries"""
def resolve(
    raw: LLMOutput,
    question: str,
    today: date,
    prior: Optional[DomainEntities] = None,
    max_queries: int = 4,
) -> Result:
    if raw.domain == TaxDomain.OUT_OF_SCOPE or raw.intent == Intent.OUT_OF_SCOPE:
        return Refuse(reason="This assistant only handles Singapore tax matters under IRAS jurisdiction (CIT, Property Tax, GST, Stamp Duty, Individual Tax).")

    active_entities = get_active_entities(raw)
    merged = merge_domain_entities(prior, active_entities)
    entities, tax_year, assumptions = normalise_entities(raw.domain, merged, today)

    # 1. Deterministic Missing Slots Check
    required_keys = MANDATORY_SLOTS.get((raw.domain, raw.intent), [])
    missing = [k for k in required_keys if getattr(entities, k, None) is None]

    # Contextual check: SUTE eligibility requires incorporation timeframe
    if raw.domain == TaxDomain.CIT and raw.intent == Intent.ELIGIBILITY and isinstance(entities, CITEntities):
        is_sute = any("start" in t.lower() or "sute" in t.lower() for t in entities.topics)
        if is_sute and entities.is_new_company is None and entities.incorporation_year is None:
            missing.append("incorporation_year")

    if missing:
        questions = [CLARIFY_TEMPLATES[m] for m in missing if m in CLARIFY_TEMPLATES]
        return Clarify(
            question=" ".join(questions),
            missing=missing,
            domain=raw.domain,
            entities=entities,
        )

    # 2. Build Sub-Queries with Metadata Filters
    queries: List[PlannedQuery] = []
    seen = set()

    for sq in raw.sub_queries:
        txt = sq.query.strip()
        if txt and txt.lower() not in seen:
            seen.add(txt.lower())
            queries.append(PlannedQuery(
                text=txt,
                filter=SearchFilter(domain=raw.domain, year=tax_year, doc_type=sq.doc_type),
            ))

    if not queries:
        queries.append(PlannedQuery(
            text=question.strip(),
            filter=SearchFilter(domain=raw.domain, year=tax_year, doc_type=None),
        ))

    return Decompose(
        domain=raw.domain,
        intent=raw.intent,
        entities=entities,
        tax_year=tax_year,
        assumptions=assumptions,
        queries=queries[:max_queries],
    )


def intake_and_plan(
    messages: list[dict],
    today: Optional[date] = None,
    prior: Optional[DomainEntities] = None,
    model: str = MODEL,
) -> Result:
    today = today or date.today()
    question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    raw = extract(messages, today, model)
    return resolve(raw, question, today, prior=prior)


# ------------------------------------------------------------------ 4. Quick Terminal Test

if __name__ == "__main__":
    conversation: list[dict] = []
    prior_state: Optional[DomainEntities] = None

    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        conversation.append({"role": "user", "content": q})
        raw = extract(conversation, date.today(), MODEL)
        res = resolve(raw, q, date.today(), prior=prior_state)

        if isinstance(res, Clarify):
            print(f"bot [Clarify]> {res.question}")
            conversation.append({"role": "assistant", "content": res.question})
            prior_state = res.entities
        else:
            print(f"bot [Decomposed Plan]:\n{res.model_dump_json(indent=2)}")
            conversation, prior_state = [], None