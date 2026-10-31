"""Intake gate and query planner.

Two separate jobs, called at different points in the pipeline:

  intake_and_plan(messages, today, prior)      runs BEFORE the supervisor
      extract()  one small LLM call: domain, intent, entities (no sub-queries)
      resolve()  deterministic gates: Refuse (out of scope), Clarify (missing mandatory slot), normalisation
                 -> Refuse | Clarify | Decompose(planned=False, queries=[the question itself])

  plan_queries(question, plan, missing, tried)  called BY the supervisor, only when it decides the question needs it
      one LLM call that breaks the question (or the fact the auditor says is missing) into keyword sub-queries

`intake_and_plan(..., plan_queries=True)` restores the original single-call behaviour (extraction and sub-queries
together, Decompose(planned=True)), so the two pipelines can be compared on the same eval set.
"""
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
    IntakeOutput,
    LLMOutput,
    QueryPlan,
    SubQueryDraft,
    Clarify,
    Refuse,
    Decompose,
    PlannedQuery,
    SearchFilter,
    Result,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable when run on its own
from src.config import INTAKE_MODEL as MODEL, PLANNER_MODEL    # the models live in src/config.py
NUM_CTX = 8192

FEW_SHOT_PATH = Path(__file__).parent / "few_shots.txt"

# ------------------------------------------------------------------ 1. Few-shots
def load_few_shot_examples(filepath: Path | str = FEW_SHOT_PATH) -> list[list[dict]]:
    """Parses few_shots.txt into one list of role-based messages per example."""
    path = Path(filepath)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    examples: list[list[dict]] = []
    for block in content.split("=== EXAMPLE "):
        if not block.strip():
            continue
        messages: list[dict] = []
        current_role = None
        current_content: list[str] = []

        def flush():
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})

        for line in block.strip().splitlines():
            if line.startswith("==="):
                continue
            if line.strip() in ("<|user|>", "<|assistant|>"):
                flush()
                current_content = []
                current_role = "user" if line.strip() == "<|user|>" else "assistant"
            elif current_role:
                current_content.append(line)
        flush()
        if messages:
            examples.append(messages)
    return examples


def load_few_shots_from_txt(filepath: Path | str = FEW_SHOT_PATH) -> list[dict]:
    """All examples as one flat message list (the original format, with sub-queries)."""
    return [m for ex in load_few_shot_examples(filepath) for m in ex]


def _json_or_none(text: str) -> Optional[dict]:
    try:
        v = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def _intake_shots(examples: list[list[dict]]) -> list[dict]:
    """The examples with `sub_queries` removed from the assistant JSON: intake only extracts."""
    out = []
    for ex in examples:
        for m in ex:
            data = _json_or_none(m["content"]) if m["role"] == "assistant" else None
            if data is not None and "sub_queries" in data:
                data = {k: v for k, v in data.items() if k != "sub_queries"}
                m = {"role": "assistant", "content": json.dumps(data)}
            out.append(m)
    return out


def _planner_shots(examples: list[list[dict]]) -> list[dict]:
    """Single-turn examples that have sub-queries, rewritten as question -> {"sub_queries": [...]}."""
    out = []
    for ex in examples:
        if len(ex) != 2 or ex[0]["role"] != "user":
            continue
        data = _json_or_none(ex[1]["content"])
        if not data or not data.get("sub_queries"):
            continue
        out += [{"role": "user", "content": f"QUESTION: {ex[0]['content']}"},
                {"role": "assistant", "content": json.dumps({"sub_queries": data["sub_queries"]})}]
    return out


FEW_SHOT_EXAMPLES = load_few_shot_examples(FEW_SHOT_PATH)
FEW_SHOT_MESSAGES = [m for ex in FEW_SHOT_EXAMPLES for m in ex]       # original: extraction + sub-queries
INTAKE_SHOT_MESSAGES = _intake_shots(FEW_SHOT_EXAMPLES)               # extraction only
PLANNER_SHOT_MESSAGES = _planner_shots(FEW_SHOT_EXAMPLES)             # sub-queries only


# ------------------------------------------------------------------ 2. Intake extraction
def build_system_prompt(today: date, plan_queries: bool = False) -> str:
    planning_rule = ("   - Generate 1-3 targeted sub-queries aimed at IRAS regulations, rates, or circulars.\n"
                     if plan_queries else
                     "   - Do NOT write search queries; only classify the request and extract its facts.\n")
    return f"""You are the intake stage of an official Singapore IRAS Tax assistant.
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
{planning_rule}
"""


def _chat(model: str, messages: list[dict], schema: dict) -> str:
    import ollama
    kwargs = dict(model=model, format=schema, options={"temperature": 0, "num_ctx": NUM_CTX})
    try:
        resp = ollama.chat(messages=messages, think=False, **kwargs)
    except TypeError:                                   # older client without `think`
        resp = ollama.chat(messages=messages, **kwargs)
    return resp["message"]["content"]


def extract(messages: list[dict], today: date, model: str = MODEL, plan_queries: bool = False):
    """IntakeOutput (default) or, with plan_queries=True, LLMOutput with sub-queries (the original single call)."""
    schema_cls = LLMOutput if plan_queries else IntakeOutput
    shots = FEW_SHOT_MESSAGES if plan_queries else INTAKE_SHOT_MESSAGES
    msgs = [{"role": "system", "content": build_system_prompt(today, plan_queries)}, *shots, *messages]
    last_err: Optional[Exception] = None

    for _ in range(2):
        content = _chat(model, msgs, schema_cls.model_json_schema())
        if os.environ.get("INTAKE_DEBUG"):
            print(f"[intake raw model output]\n{content}\n", flush=True)
        try:
            return schema_cls.model_validate_json(content)
        except ValidationError as err:
            last_err = err
            msgs += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Schema validation failed: {err}. Return corrected JSON only."}
            ]
    raise RuntimeError(f"Intake model failed schema validation twice: {last_err}")


# ------------------------------------------------------------------ 3. Deterministic Verification & Requirements

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


def get_active_entities(raw: IntakeOutput) -> DomainEntities:
    mapping = {
        TaxDomain.CIT: (raw.cit, CITEntities),
        TaxDomain.PROPERTY_TAX: (raw.property_tax, PropertyTaxEntities),
        TaxDomain.GST: (raw.gst, GSTEntities),
        TaxDomain.STAMP_DUTY: (raw.stamp_duty, StampDutyEntities),
        TaxDomain.INDIVIDUAL_TAX: (raw.individual_tax, IndividualTaxEntities),
    }
    obj, fallback_cls = mapping.get(raw.domain, (None, CITEntities))
    return obj if obj is not None else fallback_cls()


# ------------------------------------------------------------------ 4. Gates & Entry Point

def resolve(
    raw: IntakeOutput,
    question: str,
    today: date,
    prior: Optional[DomainEntities] = None,
    max_queries: int = 4,
    search_text: Optional[str] = None,
) -> Result:
    """The deterministic gates. Never splits the user's prompt into more than `max_queries` queries.

    raw: IntakeOutput (no sub-queries) -> Decompose(planned=False) whose one query is `search_text` (or the question);
         LLMOutput with sub-queries    -> Decompose(planned=True) with those queries (the original behaviour).
    search_text: what to search when nothing was planned; intake_and_plan passes every user turn of the conversation,
         so a clarification answer such as "120,000" is searched together with the question it answers.
    """
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

    # 2. Sub-queries, only when the intake call was asked to plan them
    queries = build_planned_queries(getattr(raw, "sub_queries", []), raw.domain, tax_year, max_queries)
    planned = bool(queries)
    if not queries:
        queries.append(PlannedQuery(
            text=(search_text or question).strip(),
            filter=SearchFilter(domain=raw.domain, year=tax_year),
        ))

    return Decompose(
        domain=raw.domain,
        intent=raw.intent,
        entities=entities,
        tax_year=tax_year,
        assumptions=assumptions,
        queries=queries[:max_queries],
        planned=planned,
    )


def build_planned_queries(drafts: list, domain: TaxDomain, year: Optional[int], max_queries: int = 4) -> list[PlannedQuery]:
    """SubQueryDrafts -> PlannedQuery with the plan's filter; blanks and case-insensitive repeats dropped."""
    queries: List[PlannedQuery] = []
    seen = set()
    for sq in drafts:
        txt = re.sub(r"\s+", " ", sq.query).strip()
        if txt and txt.lower() not in seen:
            seen.add(txt.lower())
            queries.append(PlannedQuery(text=txt, filter=SearchFilter(domain=domain, year=year)))
    return queries[:max_queries]


def _search_text(messages: list[dict]) -> str:
    """Every user turn of the conversation, oldest first: a clarification answer alone ("120,000") searches badly."""
    turns = [m["content"].strip() for m in messages if m["role"] == "user" and m["content"].strip()]
    return " ".join(turns)


def intake_and_plan(
    messages: list[dict],
    today: Optional[date] = None,
    prior: Optional[DomainEntities] = None,
    model: str = MODEL,
    plan_queries: bool = False,
) -> Result:
    """Intake gate. plan_queries=False (default): extraction only, the supervisor decides about decomposition.
    plan_queries=True: the original pipeline, sub-queries written in the same call."""
    today = today or date.today()
    question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    raw = extract(messages, today, model, plan_queries=plan_queries)
    return resolve(raw, question, today, prior=prior, search_text=_search_text(messages))


# ------------------------------------------------------------------ 5. Query planner (the supervisor's decompose step)

PLANNER_SYSTEM = """You plan searches for a Singapore IRAS tax assistant.
Break the QUESTION into 1-{max_queries} short keyword search queries, ONE fact or rule per query, using official IRAS
terms (e.g. "partial tax exemption", "start-up tax exemption qualifying conditions", "Form C-S filing due date").
- If MISSING is given, the earlier searches already cover the rest: write queries ONLY for the missing fact.
- Never repeat or rephrase a query listed under ALREADY SEARCHED.
- Do not put the user's own figures in a query unless a rate table is keyed on them.
Reply with JSON: {{"sub_queries": [{{"query": "...", "doc_type": "..."}}]}}"""


def build_planner_messages(question: str, known: str = "", missing: Optional[str] = None, tried=(),
                           max_queries: int = 3) -> list[dict]:
    user = f"QUESTION: {question}"
    if known:
        user += f"\nKNOWN: {known}"
    if missing:
        user += f"\nMISSING: {missing}"
    if tried:
        user += "\nALREADY SEARCHED:\n" + "\n".join(f"- {t}" for t in tried)
    return [{"role": "system", "content": PLANNER_SYSTEM.format(max_queries=max_queries)},
            *PLANNER_SHOT_MESSAGES,
            {"role": "user", "content": user}]


def plan_queries(question: str, known: str = "", missing: Optional[str] = None, tried=(), max_queries: int = 3,
                 model: str = PLANNER_MODEL) -> list[str]:
    """The sub-queries for `question` (or only for the `missing` fact). Returns [] if the model output is unusable,
    so the caller can fall back to searching the question as it is."""
    msgs = build_planner_messages(question, known, missing, tried, max_queries)
    try:
        plan = QueryPlan.model_validate_json(_chat(model, msgs, QueryPlan.model_json_schema()))
    except (ValidationError, ValueError):
        return []
    out, seen = [], set()
    for sq in plan.sub_queries:
        txt = re.sub(r"\s+", " ", sq.query).strip()
        if txt and txt.lower() not in seen:
            seen.add(txt.lower())
            out.append(txt)
    return out[:max_queries]


# ------------------------------------------------------------------ 6. Quick Terminal Test

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
        res = intake_and_plan(conversation, date.today(), prior=prior_state)

        if isinstance(res, Clarify):
            print(f"bot [Clarify]> {res.question}")
            conversation.append({"role": "assistant", "content": res.question})
            prior_state = res.entities
        else:
            print(f"bot [Intake result]:\n{res.model_dump_json(indent=2)}")
            if isinstance(res, Decompose):
                print("planner sub-queries:", plan_queries(res.queries[0].text))
            conversation, prior_state = [], None