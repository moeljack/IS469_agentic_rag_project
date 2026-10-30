"""Offline tests for the slim intake gate + the supervisor's route / decompose steps. No Ollama needed."""
import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
for _d in ("inference", "decomposer"):
    sys.path.insert(0, str(_ROOT / "src" / _d))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import decomposer
from decomposer import (CITEntities, Clarify, Decompose, IntakeOutput, Intent, LLMOutput, PlannedQuery, Refuse,
                        SearchFilter, TaxDomain, resolve)
from supervisor import Supervisor, ROUTE_SCHEMA, DECIDE_SCHEMA
from test_supervisor import (C, D, FakeRetrieval, ScriptedLLM, judge_script, store, sufficient_judge)  # noqa: F401
from supervisor import Auditor

TODAY = date(2026, 10, 20)


def slim(question="what is the rate?", intent=Intent.LOOKUP, **ent):
    """What the slim intake gate returns for an in-scope question with nothing missing."""
    return resolve(IntakeOutput(domain=TaxDomain.CIT, intent=intent, cit=CITEntities(ya=2026, **ent)), question, TODAY)


class Planner:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, question, known, missing, tried, max_queries):
        self.calls.append({"question": question, "known": known, "missing": missing, "tried": list(tried),
                           "max_queries": max_queries})
        r = self.replies.pop(0) if self.replies else []
        if isinstance(r, Exception):
            raise r
        return r


def make(store, decisions=(), batches=((C(1),),), judge=None, planner=None, max_steps=4, **kw):
    llm = ScriptedLLM(decisions)
    ret = FakeRetrieval(*batches)
    planner = planner or Planner()
    sup = Supervisor(store, ret, Auditor(judge=judge or judge_script(*[({1, 2, 3}, [])] * 6)), llm=llm,
                     planner=planner, max_steps=max_steps, **kw)
    return sup, llm, ret, planner


# ---------------------------------------------------------------- intake gate (decomposer.resolve)
def test_slim_intake_returns_an_unplanned_decompose_whose_query_is_the_question():
    p = slim("what is the corporate tax rate?")
    assert isinstance(p, Decompose) and p.planned is False
    assert [q.text for q in p.queries] == ["what is the corporate tax rate?"]
    assert p.queries[0].filter.domain == TaxDomain.CIT and p.queries[0].filter.year == 2026


def test_intake_with_sub_queries_is_still_the_original_planned_pipeline():
    raw = LLMOutput(domain=TaxDomain.CIT, intent=Intent.LOOKUP, cit=CITEntities(ya=2026),
                    sub_queries=[{"query": "cit rate"}, {"query": "CIT  rate"}, {"query": "cit rebate"}])
    p = resolve(raw, "q", TODAY)
    assert p.planned is True and [q.text for q in p.queries] == ["cit rate", "cit rebate"]


def test_gates_still_refuse_and_clarify_before_any_supervisor_work():
    assert isinstance(resolve(IntakeOutput(domain=TaxDomain.OUT_OF_SCOPE, intent=Intent.OUT_OF_SCOPE), "q", TODAY),
                      Refuse)
    clar = resolve(IntakeOutput(domain=TaxDomain.CIT, intent=Intent.COMPUTATION, cit=CITEntities(ya=2026)), "q", TODAY)
    assert isinstance(clar, Clarify) and clar.missing == ["chargeable_income"]


def test_refused_question_never_reaches_the_route_step(store):
    sup, llm, ret, planner = make(store)
    out = sup.supervise("u1", resolve(IntakeOutput(domain=TaxDomain.OUT_OF_SCOPE, intent=Intent.OUT_OF_SCOPE),
                                      "best stock to buy?", TODAY), "best stock to buy?")
    assert out.status == "refused" and llm.decide_calls == [] and ret.calls == [] and planner.calls == []


def test_clarification_answer_is_searched_together_with_the_question_it_answers():
    p = resolve(IntakeOutput(domain=TaxDomain.CIT, intent=Intent.COMPUTATION, cit=CITEntities(chargeable_income=120000)),
                "120,000", TODAY, search_text=decomposer._search_text([
                    {"role": "user", "content": "Can you calculate our corporate tax payable?"},
                    {"role": "assistant", "content": "What is the chargeable income?"},
                    {"role": "user", "content": "120,000"}]))
    assert p.queries[0].text == "Can you calculate our corporate tax payable? 120,000"


# ---------------------------------------------------------------- route step
def test_simple_question_routes_to_search_as_written_without_the_planner(store):
    sup, llm, ret, planner = make(store, [D("search"), D("answer")])
    out = sup.supervise("u1", slim("what is the corporate tax rate?"), "what is the corporate tax rate?")
    assert out.route == "search" and out.status == "answered" and planner.calls == []
    assert ret.calls[0][0] == ["what is the corporate tax rate?"]
    assert llm.decide_calls[0][0]["content"].startswith("You route questions")      # step 1 used the route prompt
    assert "route" in dict(out.timings) and "plan" not in dict(out.timings)


def test_complex_question_routes_to_decompose_and_searches_the_planner_queries(store):
    planner = Planner(["start-up tax exemption qualifying conditions", "partial tax exemption",
                       "corporate income tax rebate"])
    sup, llm, ret, _ = make(store, [D("decompose"), D("answer")], planner=planner)
    q = "Do we qualify for SUTE, and if not what is our tax with PTE and the rebate?"
    out = sup.supervise("u1", slim(q, intent=Intent.ELIGIBILITY, is_new_company=True), q)
    assert out.route == "decompose" and out.status == "answered"
    assert planner.calls[0]["question"] == q and planner.calls[0]["missing"] is None
    assert "newly incorporated: True" in planner.calls[0]["known"] and planner.calls[0]["max_queries"] == 2
    assert ret.calls[0][0] == ["start-up tax exemption qualifying conditions", "partial tax exemption"]  # max_queries
    assert any(t.startswith("  planner queries: ") for t in out.trace) and "plan" in dict(out.timings)


def test_route_sees_intent_and_known_facts(store):
    sup, llm, _, _ = make(store, [D("search"), D("answer")])
    sup.supervise("u1", slim("tax rate?", fye="2025-12-31"), "tax rate?")
    state = llm.decide_calls[0][1]["content"]
    assert "QUESTION: tax rate?" in state and "INTENT: lookup" in state and "financial year end: 2025-12-31" in state


@pytest.mark.parametrize("bad", ["not json", json.dumps({"action": "answer", "reason": "x"})])
def test_unusable_route_output_falls_back_to_searching_as_written(store, bad):
    sup, _, ret, planner = make(store, [bad, D("answer")])
    out = sup.supervise("u1", slim("rate?"), "rate?")
    assert out.route == "search" and ret.calls[0][0] == ["rate?"] and planner.calls == []
    assert any("route unparseable" in t for t in out.trace)


@pytest.mark.parametrize("reply", [[], RuntimeError("ollama down")])
def test_planner_failure_at_step_1_still_searches_the_question(store, reply):
    sup, _, ret, _ = make(store, [D("decompose"), D("answer")], planner=Planner(reply))
    out = sup.supervise("u1", slim("rate?"), "rate?")
    assert ret.calls[0][0] == ["rate?"] and out.status == "answered"
    assert any("falling back to the question as written" in t for t in out.trace)


def test_planned_intake_skips_the_route_call(store):
    raw = LLMOutput(domain=TaxDomain.CIT, intent=Intent.LOOKUP, cit=CITEntities(ya=2026),
                    sub_queries=[{"query": "corporate income tax rate"}])
    sup, llm, ret, planner = make(store, [D("answer")])
    out = sup.supervise("u1", resolve(raw, "rate?", TODAY), "rate?")
    assert out.route == "planned" and len(llm.decide_calls) == 1 and ret.calls[0][0] == ["corporate income tax rate"]
    assert "route" not in dict(out.timings)


def test_sufficient_first_search_still_answers_with_just_the_route_call(store):
    sup, llm, ret, _ = make(store, [D("search")], judge=sufficient_judge(True))
    out = sup.supervise("u1", slim("rate?"), "rate?")
    assert out.stop_reason == "auditor_sufficient" and len(llm.decide_calls) == 1 and len(ret.calls) == 1


# ---------------------------------------------------------------- decompose mid-loop: re-plan for the missing fact
def test_later_decompose_plans_only_for_what_the_auditor_says_is_missing(store):
    def judge(q, chunks, kept, tried=(), year=None):
        return {"chunks": [{"index": 1, "relevant": True, "reason": ""}], "suggested_queries": [],
                "sufficient": False, "missing": "the shareholder count limit"}
    planner = Planner(["start-up tax exemption number of shareholders"])
    sup, llm, ret, _ = make(store, [D("search"), D("decompose"), D("answer")], batches=([C(1)], [C(2)]),
                            judge=judge, planner=planner)
    out = sup.supervise("u1", slim("Do we qualify for SUTE?"), "Do we qualify for SUTE?")
    assert planner.calls[0]["missing"] == "the shareholder count limit"
    assert planner.calls[0]["tried"] == ["Do we qualify for SUTE?"]
    assert ret.calls[1][0] == ["start-up tax exemption number of shareholders"]
    assert any("for missing 'the shareholder count limit'" in t for t in out.trace)
    assert "decompose" in DECIDE_SCHEMA["properties"]["action"]["enum"]


def test_later_decompose_drops_repeats_and_falls_back_to_auditor_suggestions(store):
    planner = Planner(["what is the rate"])                           # repeats the step-1 search
    sup, _, ret, _ = make(store, [D("search"), D("decompose"), D("answer")], batches=([C(1)], [C(2)]),
                          judge=judge_script(({1}, ["late payment penalty"]), ({1}, [])), planner=planner)
    out = sup.supervise("u1", slim("what is the rate?"), "what is the rate?")
    assert ret.calls[1][0] == ["late payment penalty"]
    assert any("skipped near-duplicate planned queries" in t for t in out.trace)


def test_later_decompose_with_nothing_new_stops_and_answers(store):
    sup, _, ret, _ = make(store, [D("search"), D("decompose")], batches=([C(1)],),
                          judge=judge_script(({1}, [])), planner=Planner([]))
    out = sup.supervise("u1", slim("rate?"), "rate?")
    assert len(ret.calls) == 1 and out.stop_reason == "no_new_queries" and out.status == "answered"


# ---------------------------------------------------------------- the real decomposer functions, Ollama faked
@pytest.fixture
def fake_ollama(monkeypatch):
    mod = ModuleType("ollama")
    mod.calls, mod.replies = [], []

    def chat(**kw):
        mod.calls.append(kw)
        return {"message": {"content": mod.replies.pop(0)}}
    mod.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", mod)
    return mod


def test_extract_asks_for_no_sub_queries_and_shows_no_sub_queries_in_its_examples(fake_ollama):
    fake_ollama.replies = [json.dumps({"domain": "cit", "intent": "lookup", "cit": {"ya": 2026}})]
    raw = decomposer.extract([{"role": "user", "content": "rate?"}], TODAY)
    call = fake_ollama.calls[0]
    assert isinstance(raw, IntakeOutput) and not hasattr(raw, "sub_queries")
    assert "sub_queries" not in json.dumps(call["format"])
    assert not any("sub_queries" in m["content"] for m in call["messages"])
    assert "Do NOT write search queries" in call["messages"][0]["content"]


def test_extract_with_plan_queries_is_the_original_single_call(fake_ollama):
    fake_ollama.replies = [json.dumps({"domain": "cit", "intent": "lookup", "sub_queries": [{"query": "cit rate"}]})]
    raw = decomposer.extract([{"role": "user", "content": "rate?"}], TODAY, plan_queries=True)
    assert isinstance(raw, LLMOutput) and raw.sub_queries[0].query == "cit rate"
    assert "sub_queries" in json.dumps(fake_ollama.calls[0]["format"])


def test_plan_queries_prompt_carries_missing_and_tried_and_output_is_deduped(fake_ollama):
    fake_ollama.replies = [json.dumps({"sub_queries": [{"query": "SUTE  shareholders"}, {"query": "sute shareholders"},
                                                       {"query": ""}, {"query": "SUTE investment holding"}]})]
    out = decomposer.plan_queries("Do we qualify?", known="year 2026", missing="shareholder limit",
                                  tried=["sute conditions"])
    user = fake_ollama.calls[0]["messages"][-1]["content"]
    assert out == ["SUTE shareholders", "SUTE investment holding"]
    assert "MISSING: shareholder limit" in user and "- sute conditions" in user and "KNOWN: year 2026" in user
    shots = fake_ollama.calls[0]["messages"][1:-1]
    assert shots and all(json.loads(m["content"]).keys() == {"sub_queries"} for m in shots if m["role"] == "assistant")


def test_plan_queries_returns_nothing_on_bad_output(fake_ollama):
    fake_ollama.replies = ["oops"]
    assert decomposer.plan_queries("q") == []


def test_default_intake_and_harness_pass_the_intake_plans_switch(store, monkeypatch):
    seen = []
    monkeypatch.setattr(decomposer, "intake_and_plan", lambda m, today, prior, plan_queries: seen.append(plan_queries)
                        or Refuse(reason="x"))
    for flag in (False, True):
        sup = Supervisor(store, None, None, llm=lambda m, s=None: "", intake_plans=flag)
        sup.handle_turn("u1", [{"role": "user", "content": "q"}], TODAY)
    assert seen == [False, True]
    import harness
    assert harness.parse_args(["--user", "u"]).intake_plans is False
    assert harness.parse_args(["--user", "u", "--intake-plans"]).intake_plans is True
    from test_supervisor import FakeRetriever
    assert harness.build_supervisor(store=store, retriever=FakeRetriever({}), intake_plans=True).intake_plans is True