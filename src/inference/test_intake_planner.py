"""Tests for the deterministic half of Steps 1+2. No LLM needed: we feed resolve() fake model output."""
from datetime import date

from intake_planner import merge_entities, resolve
from schemas import Clarify, Entities, Intent, LLMOutput, Plan, Refuse, SubQuery

TODAY = date(2026, 10, 15)


def raw(intent, sub_queries=(), **entities):
    return LLMOutput(intent=intent, entities=Entities(**entities),
                     sub_queries=[SubQuery(query=q) for q in sub_queries])


def test_out_of_scope_is_refused():
    assert isinstance(resolve(raw(Intent.OUT_OF_SCOPE), "what is GST?", TODAY), Refuse)


def test_computation_without_income_asks_for_it():
    r = resolve(raw(Intent.COMPUTATION, ya=2026), "how much tax do I pay?", TODAY)
    assert isinstance(r, Clarify) and r.missing == ["chargeable_income"]


def test_deadline_without_fye_asks_for_it():
    r = resolve(raw(Intent.DEADLINE), "when do I file?", TODAY)
    assert isinstance(r, Clarify) and r.missing == ["fye"]


def test_ya_derived_from_fye():
    r = resolve(raw(Intent.DEADLINE, fye="2025-12-31"), "when do I file?", TODAY)
    assert isinstance(r, Plan) and r.ya == 2026 and not r.assumptions


def test_ya_derived_from_fy():
    r = resolve(raw(Intent.LOOKUP, fy=2025), "what is the rate?", TODAY)
    assert r.ya == 2026


def test_missing_ya_defaults_to_current_and_is_disclosed():
    r = resolve(raw(Intent.LOOKUP, ["corporate tax rate"]), "what is the tax rate?", TODAY)
    assert r.ya == 2026 and len(r.assumptions) == 1


def test_bad_values_are_dropped_not_trusted():
    r = resolve(raw(Intent.LOOKUP, fye="not-a-date", ya=1850), "rate?", TODAY)
    assert r.entities.fye is None and r.ya == 2026


def test_startup_eligibility_needs_incorporation_info():
    r = resolve(raw(Intent.ELIGIBILITY, topics=["Start-Up Tax Exemption"]), "do I qualify?", TODAY)
    assert isinstance(r, Clarify) and r.missing == ["incorporation"]


def test_queries_get_ya_filter_and_original_question_fallback():
    r = resolve(raw(Intent.LOOKUP, ["CIT rebate"], ya=2026), "what is the rebate?", TODAY)
    texts = [q.text for q in r.queries]
    assert texts == ["CIT rebate", "what is the rebate?"]
    assert all(q.filter.ya == 2026 for q in r.queries)


def test_duplicate_queries_removed():
    r = resolve(raw(Intent.LOOKUP, ["rate", "Rate"]), "rate", TODAY)
    assert len(r.queries) == 1


def test_clarification_turn_merges_entities():
    first = resolve(raw(Intent.COMPUTATION, ya=2026), "how much tax?", TODAY)
    assert isinstance(first, Clarify)
    second = resolve(raw(Intent.COMPUTATION, chargeable_income=300000), "300k", TODAY, prior=first.entities)
    assert isinstance(second, Plan) and second.ya == 2026 and second.entities.chargeable_income == 300000
    assert "compute_tax" in second.tools


def test_merge_entities_unions_topics():
    a = Entities(topics=["Form C-S"], ya=2026)
    b = Entities(topics=["ECI", "Form C-S"])
    m = merge_entities(a, b)
    assert m.ya == 2026 and m.topics == ["Form C-S", "ECI"]