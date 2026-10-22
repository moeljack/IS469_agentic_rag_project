"""Offline tests for the supervisor harness: no Ollama, no embedding models, in-memory ChromaDB."""
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]                    # repo root
sys.path.insert(0, str(_ROOT))
for _d in ("inference", "decomposer"):                         # decomposer, decomposer_general_schemas
    sys.path.insert(0, str(_ROOT / "src" / _d))
sys.path.insert(0, str(Path(__file__).resolve().parent))       # supervisor

import chromadb
import pytest

from decomposer import resolve
from src.decomposer.decomposer_general_schemas import (
    CITEntities, Clarify, Decompose, IndividualTaxEntities, Intent, LLMOutput, PlannedQuery, PropertyTaxEntities,
    Refuse, SearchFilter, SubQueryDraft, TaxDomain,
)
from supervisor import Auditor, Chunk, RetrievalAgent, RetrievalAudit, Supervisor, agent_judge
from src.retrieval.user_store import UserStore

TODAY = date(2026, 10, 20)
DOC = Chunk(id="web:1:0", content="Your company is taxed at a flat rate of 17% of its chargeable income.",
            metadata={"source": "https://www.iras.gov.sg/rate"})


def plan(intent=Intent.LOOKUP, **ent):
    return Decompose(domain=TaxDomain.CIT, intent=intent, entities=CITEntities(**ent), tax_year=2026,
                     queries=[PlannedQuery(text="rate", filter=SearchFilter(domain=TaxDomain.CIT, year=2026))])


class FakeRetrieval:
    """Returns the first scripted batch for the plan's queries, then one batch per call. Records queries used."""
    def __init__(self, *batches): self.batches, self.calls, self.last_notes = list(batches), [], []
    def run(self, plan, queries=None, exclude=()):
        self.calls.append(([q.text for q in (queries or plan.queries)], set(exclude)))
        return list(self.batches.pop(0)) if self.batches else []


def judge_script(*steps):
    """steps: (set of relevant 1-based indexes, suggested queries) per audit call."""
    it = iter(steps)
    def judge(question, chunks, kept):
        rel, sugg = next(it)
        return {"chunks": [{"index": i, "relevant": i in rel, "reason": "r"} for i in range(1, len(chunks) + 1)],
                "suggested_queries": sugg}
    return judge


class ScriptedLLM:
    """Stands in for qwen. Calls with a schema are supervisor decisions; calls without are the answer step."""
    def __init__(self, decisions=(), answer="The rate is 17% [Doc 1]."):
        self.decisions, self.answer, self.decide_calls, self.answer_calls = list(decisions), answer, [], []

    def __call__(self, messages, schema=None):
        if schema is None:
            self.answer_calls.append(messages)
            return self.answer
        self.decide_calls.append(messages)
        d = self.decisions.pop(0) if self.decisions else {"action": "answer", "reason": "default"}
        return d if isinstance(d, str) else json.dumps(d)

    @property
    def answered_docs(self):                       # the DOCUMENTS section the answer prompt was given
        return self.answer_calls[0][1]["content"].split("DOCUMENTS:")[1] if self.answer_calls else ""


def D(action, **kw):
    return {"action": action, "reason": "t", **kw}


@pytest.fixture
def store():
    import uuid
    return UserStore(client=chromadb.EphemeralClient(), collection_name=f"t{uuid.uuid4().hex}")


def C(n, text="x", src=None):
    return Chunk(id=f"web:{n}:0", content=text, metadata={"source": src or f"https://iras/{n}"})


def make(store, decisions=(), answer="The rate is 17% [Doc 1].", batches=((DOC,),), judge=None, max_steps=4,
         intake=None, tools=None):
    llm = ScriptedLLM(decisions, answer)
    ret = FakeRetrieval(*batches)
    judge = judge or judge_script(*[({1, 2, 3, 4, 5, 6}, [])] * 6)
    sup = Supervisor(store, ret, Auditor(judge=judge), llm=llm, intake=intake, tools=tools, max_steps=max_steps)
    return sup, llm, ret


# ---------------------------------------------------------------- adapter: Retriever -> RetrievalAgent
class FakeRetriever:
    """Stands in for retriever.Retriever: search(query, k, ...) -> hit dicts, best first."""
    def __init__(self, table): self.table, self.calls = table, []
    def search(self, query, k=5, source=None, min_score=None, candidates=None):
        self.calls.append((query, k, min_score))
        return [h for h in self.table.get(query, []) if min_score is None or h["score"] >= min_score][:k]


def H(id_, score, text="t", headings="Page > Section"):
    return {"id": id_, "score": score, "text": text, "headings": headings, "pages": "", "source_type": "webpage",
            "source": f"https://iras/{id_}", "title": "T"}


def planned(*texts):
    return [PlannedQuery(text=t, filter=SearchFilter(domain=TaxDomain.CIT, year=2026)) for t in texts]


def test_retrieval_agent_runs_each_planned_query_dedupes_and_orders_by_score():
    r = FakeRetriever({"a": [H("web:1:0", .5), H("web:2:0", .9)], "b": [H("web:2:0", .9), H("web:3:0", .7)]})
    p = plan(); p.queries = planned("a", "b")
    agent = RetrievalAgent(r, k=5)
    out = agent.run(p)
    assert [q[0] for q in r.calls] == ["a", "b"]
    assert [c.id for c in out] == ["web:2:0", "web:3:0", "web:1:0"]
    assert out[0].metadata["source"] == "https://iras/web:2:0" and out[0].metadata["headings"] == "Page > Section"
    assert out[0].content == "t" and len(agent.last_notes) == 2


def test_retrieval_agent_uses_given_queries_and_skips_excluded():
    r = FakeRetriever({"x": [H("web:1:0", .9), H("web:2:0", .8)]})
    out = RetrievalAgent(r).run(plan(), queries=planned("x"), exclude={"web:1:0"})
    assert [q[0] for q in r.calls] == ["x"] and [c.id for c in out] == ["web:2:0"]


def test_retrieval_agent_caps_batch_and_passes_min_score():
    r = FakeRetriever({"a": [H(f"web:{i}:0", 1 - i / 20) for i in range(1, 6)]})
    p = plan(); p.queries = planned("a")
    assert len(RetrievalAgent(r, max_chunks=3).run(p)) == 3
    assert len(RetrievalAgent(r, min_score=0.92).run(p)) == 1 and r.calls[-1][2] == 0.92


# ---------------------------------------------------------------- adapter: AuditorAgent -> Auditor
def fake_agent(evals, sufficient=False, recommended=None):
    calls = []
    def evaluate(original_query, sub_prompt, chunks):
        calls.append((original_query, sub_prompt, chunks))
        return SimpleNamespace(is_sufficient=sufficient, recommended_search_query=recommended,
                               chunk_evaluations=[SimpleNamespace(chunk_id=i, status=s, reason=r)
                                                  for i, s, r in evals])
    return SimpleNamespace(evaluate=evaluate, calls=calls)


def test_agent_judge_maps_chunk_ids_to_verdicts_and_passes_chunk_fields():
    agent = fake_agent([("web:1:0", "approved", "has rate"), ("web:2:0", "rejected", "off topic")],
                       recommended="late payment penalty")
    a, b = C(1, "alpha"), C(2, "bravo")
    a.metadata["headings"] = "Rates"
    audit = Auditor(judge=agent_judge(agent)).audit("q", [a, b], "- Earlier topic")
    assert audit.relevant == [True, False] and audit.reasons == ["has rate", "off topic"]
    assert audit.suggested_queries == ["late payment penalty"] and audit.sufficient is False
    original, sub_prompt, payload = agent.calls[0]
    assert original == "q" and "Earlier topic" in sub_prompt
    assert payload == [{"id": "web:1:0", "headings": "Rates", "text": "alpha"},
                       {"id": "web:2:0", "headings": "", "text": "bravo"}]


def test_agent_judge_keeps_chunks_the_model_skipped():
    agent = fake_agent([("web:1:0", "rejected", "no")])
    audit = Auditor(judge=agent_judge(agent)).audit("q", [C(1), C(2)])
    assert audit.relevant == [False, True]


def test_audit_not_all_relevant_when_auditor_says_insufficient():
    assert not RetrievalAudit(relevant=[True, True], sufficient=False).all_relevant
    assert RetrievalAudit(relevant=[True, True], sufficient=True).all_relevant
    assert RetrievalAudit(relevant=[True, True]).all_relevant
    assert not RetrievalAudit(relevant=[True, False]).all_relevant


def test_auditor_fails_open_when_judge_raises():
    def broken(*a): raise RuntimeError("ollama down")
    audit = Auditor(judge=broken).audit("q", [C(1), C(2)])
    assert audit.relevant == [True, True] and audit.unavailable


# ---------------------------------------------------------------- user store
def test_profile_created_once(store):
    _, created = store.ensure_profile("u1")
    _, again = store.ensure_profile("u1")
    assert created and not again


def test_remember_persists_company_facts_only_and_logs_history(store):
    changes = store.remember("u1", CITEntities(fye="2025-12-31", chargeable_income=300000, ya=2026), "q1")
    assert changes == [("fye", None, "2025-12-31")]
    assert store.profile_entities("u1").fye == "2025-12-31"
    assert store.profile_entities("u1").chargeable_income is None        # per-question fact is not a default
    assert {h["field"] for h in store.history("u1")} == {"fye", "chargeable_income", "ya"}


def test_changed_value_overwrites_but_history_keeps_old(store):
    store.remember("u1", CITEntities(fye="2025-12-31"), "a")
    changes = store.remember("u1", CITEntities(fye="2025-06-30"), "b")
    assert changes == [("fye", "2025-12-31", "2025-06-30")]
    assert [h["value"] for h in store.history("u1", "fye")] == ["2025-12-31", "2025-06-30"]


def test_users_are_isolated_and_forget_deletes(store):
    store.remember("u1", CITEntities(fye="2025-12-31"), "a")
    assert store.profile_entities("u2").fye is None
    store.forget("u1")
    assert store.get_profile("u1") is None and store.history("u1") == []


def test_company_info_rejects_unknown_fields(store):
    store.set_company_info("u1", company_name="Acme Pte Ltd", uen="201912345A")
    assert store.get_profile("u1")["company"]["uen"] == "201912345A"
    with pytest.raises(ValueError):
        store.set_company_info("u1", password="x")


# ---------------------------------------------------------------- agent loop
def test_first_step_is_planned_search_then_model_answers_and_supervisor_writes_it(store):
    sup, llm, ret = make(store, [D("answer")])
    out = sup.supervise("u1", plan(), "what is the rate?")
    assert out.status == "answered" and out.stop_reason == "agent_answered"
    assert len(ret.calls) == 1 and len(llm.decide_calls) == 1 and len(llm.answer_calls) == 1
    assert "https://www.iras.gov.sg/rate" in out.message and out.message.startswith("The rate is 17%")


def test_model_searches_again_with_its_own_queries_and_unrelated_chunks_never_reach_the_answer(store):
    a, b, good = C(1, "alpha"), C(2, "bravo"), C(3, "good")
    sup, llm, ret = make(store, [D("search", queries=["better query"]), D("answer")], batches=([a, b], [good]),
                         judge=judge_script(({1}, ["auditor idea"]), ({1}, [])))
    out = sup.supervise("u1", plan(), "q")
    assert ret.calls[1][0] == ["better query"] and ret.calls[1][1] == {a.id, b.id}
    assert "alpha" in llm.answered_docs and "good" in llm.answered_docs and "bravo" not in llm.answered_docs
    assert len(out.audits) == 2 and out.status == "answered"


def test_search_without_queries_falls_back_to_auditor_suggestions(store):
    sup, _, ret = make(store, [D("search"), D("answer")], batches=([C(1), C(2)], [C(3)]),
                       judge=judge_script(({1}, ["suggested by auditor"]), ({1}, [])))
    sup.supervise("u1", plan(), "q")
    assert ret.calls[1][0] == ["suggested by auditor"]


def test_search_with_no_planned_queries_still_builds_valid_follow_up_queries(store):
    p = plan(); p.queries = []
    sup, _, ret = make(store, [D("search", queries=["new one"]), D("answer")], batches=([C(1)], [C(2)]),
                       judge=judge_script(({1}, []), ({1}, [])))
    sup.supervise("u1", p, "q")
    assert ret.calls[1][0] == ["new one"]


def test_repeated_or_missing_queries_stop_the_search(store):
    sup, _, ret = make(store, [D("search", queries=["rate"])], batches=([C(1)],), judge=judge_script(({1}, [])))
    out = sup.supervise("u1", plan(), "q")                  # "rate" was already the planned query
    assert out.stop_reason == "no_new_queries" and len(ret.calls) == 1 and out.status == "answered"


def test_budget_forces_answer_and_flags_partial_coverage(store):
    sup, llm, _ = make(store, [D("search", queries=["s1"]), D("search", queries=["s2"])], max_steps=3,
                       batches=([C(1), C(2)], [C(3), C(4)], [C(5), C(6)]),
                       judge=judge_script(({1}, ["x"]), ({1}, ["y"]), ({1}, [])))
    out = sup.supervise("u1", plan(), "q")
    assert out.stop_reason == "max_steps" and out.status == "answered"
    assert "COVERAGE NOTE" in llm.answer_calls[0][1]["content"]


def test_no_coverage_note_when_all_related(store):
    sup, llm, _ = make(store, [D("answer")])
    sup.supervise("u1", plan(), "q")
    assert "COVERAGE NOTE" not in llm.answer_calls[0][1]["content"]


def test_coverage_note_when_auditor_says_chunks_are_insufficient(store):
    judge = lambda q, chunks, kept: {"chunks": [{"index": 1, "relevant": True, "reason": "r"}],
                                     "suggested_queries": [], "sufficient": False}
    sup, llm, _ = make(store, [D("answer")], judge=judge)
    sup.supervise("u1", plan(), "q")
    assert "COVERAGE NOTE" in llm.answer_calls[0][1]["content"]


def test_answer_with_nothing_vetted_becomes_give_up_without_answer_call(store):
    sup, llm, _ = make(store, [D("answer")], batches=([C(1)],), judge=judge_script((set(), [])))
    out = sup.supervise("u1", plan(), "q")
    assert out.status == "no_answer" and llm.answer_calls == []


def test_give_up_and_no_chunks_do_not_call_answer_model(store):
    sup, llm, _ = make(store, [D("give_up")], batches=([C(1)],), judge=judge_script((set(), ["z"])))
    assert sup.supervise("u1", plan(), "q").stop_reason == "agent_gave_up" and llm.answer_calls == []
    sup, llm, _ = make(store, batches=())
    assert sup.supervise("u1", plan(), "q").status == "no_answer" and llm.answer_calls == []


def test_garbage_decision_falls_back_to_answer_when_something_vetted(store):
    sup, llm, _ = make(store, ["not json at all"])
    out = sup.supervise("u1", plan(), "q")
    assert out.status == "answered" and any("unparseable" in t for t in out.trace)


def test_unknown_action_falls_back(store):
    sup, _, _ = make(store, [{"action": "fly", "reason": "x"}])
    assert sup.supervise("u1", plan(), "q").status == "answered"


def test_ask_user_returns_clarify_and_saves_pending(store):
    sup, llm, _ = make(store, [D("ask_user", question="Which FYE?")])
    out = sup.supervise("u1", plan(chargeable_income=5), "q")
    assert out.status == "clarify" and out.message == "Which FYE?" and llm.answer_calls == []
    assert store.pop_pending("u1").chargeable_income == 5


def test_tool_runs_for_computation_only_if_registered_and_output_is_a_document(store):
    p = plan(intent=Intent.COMPUTATION, chargeable_income=300000)
    sup, llm, _ = make(store, [D("run_tool", tool="compute_tax"), D("run_tool", tool="missing_tool"),
                               D("run_tool", tool="compute_tax"), D("answer")],
                       tools={"compute_tax": lambda plan: "Tax payable: 25,200"}, max_steps=5)
    out = sup.supervise("u1", p, "tax?")
    assert out.status == "answered" and "Tax payable: 25,200" in llm.answered_docs


def test_tools_are_not_offered_for_non_computation_questions(store):
    ran = []
    sup, llm, _ = make(store, [D("run_tool", tool="compute_tax"), D("answer")],
                       tools={"compute_tax": lambda plan: ran.append(1) or "x"})
    sup.supervise("u1", plan(intent=Intent.LOOKUP), "rate?")
    assert ran == [] and "AVAILABLE TOOLS: none" in llm.decide_calls[0][1]["content"]


def test_answer_prompt_carries_user_facts_and_assumptions(store):
    p = plan(fye="2025-12-31", chargeable_income=300000); p.assumptions = ["Assumed resident company."]
    sup, llm, _ = make(store, [D("answer")])
    out = sup.supervise("u1", p, "q")
    prompt = llm.answer_calls[0][1]["content"]
    assert "financial year end: 2025-12-31" in prompt and "Assumed resident company." in prompt
    assert "Assumptions:" in out.message


def test_answer_prompt_shows_chunk_headings_when_present(store):
    doc = Chunk(id="web:9:0", content="17% flat", metadata={"source": "https://iras/r", "headings": "Rates > CIT"})
    sup, llm, _ = make(store, [D("answer")], batches=([doc],))
    sup.supervise("u1", plan(), "q")
    assert "Rates > CIT" in llm.answered_docs


def test_judge_failure_fails_open_and_is_traced(store):
    def broken(*a): raise RuntimeError("ollama down")
    sup, _, _ = make(store, [D("answer")], judge=broken)
    out = sup.supervise("u1", plan(), "q")
    assert out.status == "answered" and any("unavailable" in t for t in out.trace)


def test_insufficient_context_is_not_presented_as_answer(store):
    sup, _, _ = make(store, [D("answer")], answer="INSUFFICIENT_CONTEXT: The provided documents do not contain this.")
    assert sup.supervise("u1", plan(), "rate?").status == "no_answer"


def test_answer_model_error_is_reported_not_raised(store):
    sup, llm, _ = make(store, [D("answer")])
    def boom(messages, schema=None):
        if schema is None: raise RuntimeError("down")
        return json.dumps(D("answer"))
    sup.llm = boom
    assert sup.supervise("u1", plan(), "q").status == "no_answer"


def test_clarify_saves_pending_and_refuse_clears_it(store):
    sup, _, _ = make(store)
    out = sup.supervise("u1", Clarify(question="FYE?", missing=["fye"], domain=TaxDomain.CIT, entities=CITEntities(chargeable_income=5)), "q")
    assert out.status == "clarify"
    assert store.pop_pending("u1").chargeable_income == 5
    store.save_pending("u1", CITEntities(ya=2026))
    sup.supervise("u1", Refuse(reason="out of scope"), "gst?")
    assert store.pop_pending("u1") is None


def test_user_facts_are_citable_so_answer_may_use_them(store):
    sup, _, _ = make(store, answer="For your company with FYE 2025-12-31 the rate is 17% [Doc 1] [Doc 2].")
    out = sup.supervise("u1", plan(fye="2025-12-31"), "rate for me?")
    assert out.status == "answered"


def test_non_cit_answers_do_not_touch_the_company_profile_and_pending_keeps_its_domain(store):
    p = Decompose(domain=TaxDomain.INDIVIDUAL_TAX, intent=Intent.LOOKUP, tax_year=2026,
                  entities=IndividualTaxEntities(tax_resident=True, employment_income=90000),
                  queries=[PlannedQuery(text="reliefs", filter=SearchFilter(domain=TaxDomain.INDIVIDUAL_TAX, year=2026))])
    sup, llm, _ = make(store, [D("answer")])
    out = sup.supervise("u1", p, "my reliefs?")
    assert out.status == "answered" and store.profile_entities("u1").tax_resident is None
    assert "domain: individual_tax" in llm.answer_calls[0][1]["content"] and "employment income: 90000" in llm.answer_calls[0][1]["content"]
    store.save_pending("u1", PropertyTaxEntities(annual_value=40000, is_owner_occupied=True))
    got = store.pop_pending("u1")
    assert isinstance(got, PropertyTaxEntities) and got.annual_value == 40000


# ---------------------------------------------------------------- full turn
def test_handle_turn_reuses_saved_fye_and_discloses_it(store):
    store.remember("u1", CITEntities(fye="2025-12-31"), "earlier")
    seen = {}

    def intake(messages, today, prior):
        seen["prior"] = prior
        return plan(fye=prior.fye)                                        # planner would not need to ask for FYE
    sup, _, _ = make(store, [D("answer")], intake=intake)
    out = sup.handle_turn("u1", [{"role": "user", "content": "when do I file?"}], TODAY)
    assert seen["prior"].fye == "2025-12-31"
    assert out.status == "answered" and "saved company details" in out.message


def test_handle_turn_clarify_then_answer_round_trip(store):
    calls = []

    def intake(messages, today, prior):
        calls.append(prior)
        if prior.fye is None:
            return Clarify(question="FYE?", missing=["fye"], domain=TaxDomain.CIT,
                           entities=CITEntities(chargeable_income=300000))
        return plan(fye=prior.fye, chargeable_income=prior.chargeable_income)
    sup, _, _ = make(store, [D("answer")], intake=intake)
    first = sup.handle_turn("u1", [{"role": "user", "content": "tax on 300k?"}], TODAY)
    assert first.status == "clarify"
    # simulate the user answering; the pending income must come back as prior, with no fye yet
    store.remember("u1", CITEntities(fye="2025-12-31"), "31 Dec")
    second = sup.handle_turn("u1", [{"role": "user", "content": "31 Dec"}], TODAY)
    assert calls[1].chargeable_income == 300000 and second.status == "answered"


# ---------------------------------------------------------------- the real pieces wired together
def test_real_decomposer_feeds_retriever_adapter_and_auditor_adapter(store):
    """decomposer.resolve -> Supervisor -> RetrievalAgent(Retriever stand-in) -> Auditor(AuditorAgent stand-in)."""
    raw = LLMOutput(domain=TaxDomain.CIT, intent=Intent.LOOKUP, cit=CITEntities(ya=2026),
                    sub_queries=[SubQueryDraft(query="corporate income tax rate")])

    def intake(messages, today, prior):                                   # the deterministic half of decomposer
        return resolve(raw, messages[-1]["content"], today, prior)

    retriever = FakeRetriever({
        "corporate income tax rate": [H("web:1:0", .9, "Taxed at a flat rate of 17%."), H("web:2:0", .4, "Unrelated GST text")],
    })
    agent = fake_agent([("web:1:0", "approved", "gives the rate"), ("web:2:0", "rejected", "GST")], sufficient=True)
    llm = ScriptedLLM([D("answer")], "The rate is 17% [Doc 1].")
    sup = Supervisor(store, RetrievalAgent(retriever), Auditor(judge=agent_judge(agent)), llm=llm, intake=intake)

    out = sup.handle_turn("u1", [{"role": "user", "content": "what is the rate?"}], TODAY)

    assert out.status == "answered" and out.stop_reason == "agent_answered"
    assert [c[0] for c in retriever.calls] == ["corporate income tax rate"]
    assert all(q.filter.year == 2026 and q.filter.domain == TaxDomain.CIT for q in out.plan.queries)
    assert "17%" in llm.answered_docs and "Unrelated GST" not in llm.answered_docs
    assert out.sources == ["https://iras/web:1:0"]
    assert agent.calls[0][0].startswith("what is the rate?") and "year 2026" in agent.calls[0][0]


def test_real_decomposer_refuses_and_clarifies_through_the_supervisor(store):
    sup, llm, _ = make(store)
    refuse = resolve(LLMOutput(domain=TaxDomain.OUT_OF_SCOPE, intent=Intent.OUT_OF_SCOPE), "q", TODAY)
    assert sup.supervise("u1", refuse, "q").status == "refused"
    clar = resolve(LLMOutput(domain=TaxDomain.CIT, intent=Intent.COMPUTATION, cit=CITEntities(ya=2026)), "q", TODAY)
    out = sup.supervise("u1", clar, "q")
    assert out.status == "clarify" and "chargeable income" in out.message and llm.answer_calls == []
    assert store.pop_pending("u1").ya == 2026


# ---------------------------------------------------------------- speed knobs: fewer queries and chunks
def plan_with_queries(*texts):
    p = plan()
    p.queries = planned(*texts)
    return p


def test_first_search_uses_at_most_max_queries_in_order(store):
    sup, _, ret = make(store, [D("answer")], batches=([C(1)],), judge=judge_script(({1}, [])))
    sup.max_queries = 2
    out = sup.supervise("u1", plan_with_queries("q1", "q2", "q3", "q4"), "q")
    assert ret.calls[0][0] == ["q1", "q2"] and any("2 of 4 planned queries" in t for t in out.trace)


def test_follow_up_searches_are_capped_too(store):
    sup, _, ret = make(store, [D("search", queries=["a", "b", "c"]), D("answer")], batches=([C(1)], [C(2)]),
                       judge=judge_script(({1}, []), ({1}, [])))
    sup.max_queries = 2
    sup.supervise("u1", plan(), "q")
    assert ret.calls[1][0] == ["a", "b"]


def test_default_caps_are_small():
    from supervisor import DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES
    assert (DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES) == (3, 6, 2)
    assert RetrievalAgent(FakeRetriever({})).k == 3


def test_retrieval_agent_passes_candidates_only_when_set():
    seen = []

    class R:
        def search(self, query, k=5, min_score=None, **kw):
            seen.append(kw)
            return []
    p = plan(); p.queries = planned("a")
    RetrievalAgent(R()).run(p)
    RetrievalAgent(R(), candidates=12).run(p)
    assert seen == [{}, {"candidates": 12}]


# ---------------------------------------------------------------- timing in the trace
def test_handle_turn_records_stage_timings_and_a_timing_line(store):
    sup, _, _ = make(store, [D("answer")], intake=lambda m, t, prior: plan())
    out = sup.handle_turn("u1", [{"role": "user", "content": "rate?"}], TODAY)
    stages = [n for n, _ in out.timings]
    assert stages[0] == "intake" and stages[-1] == "total"
    assert {"retrieval", "audit", "decide", "answer"} <= set(stages)
    assert out.trace[-1].startswith("timing: intake ") and "total " in out.trace[-1] and "answer " in out.trace[-1]


def test_timing_line_aggregates_repeated_stages_and_reports_other():
    from supervisor import timing_line
    line = timing_line([("intake", 2.0), ("retrieval", 1.0), ("retrieval", 0.5), ("answer", 3.0), ("total", 7.0)])
    assert line == "timing: intake 2.0s | retrieval 1.5s (x2) | answer 3.0s | other 0.5s | total 7.0s"


def test_timings_are_recorded_even_when_a_stage_raises(store):
    from supervisor import _timed
    sink = []
    with pytest.raises(RuntimeError):
        _timed(sink, "audit", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert [n for n, _ in sink] == ["audit"]


def test_supervise_alone_returns_stage_timings_without_intake_or_total(store):
    sup, _, _ = make(store, [D("answer")])
    out = sup.supervise("u1", plan(), "q")
    assert [n for n, _ in out.timings][:2] == ["retrieval", "audit"] and "total" not in dict(out.timings)


# ---------------------------------------------------------------- keeping the Ollama models resident
@pytest.fixture
def fake_ollama(monkeypatch):
    import types
    mod = types.ModuleType("ollama")
    mod.calls, mod.resident = [], None

    def chat(**kw):
        mod.calls.append(kw)
        return {"message": {"content": "{}"}}

    def generate(**kw):
        mod.calls.append(("generate", kw))

    mod.chat, mod.generate = chat, generate
    mod.ps = lambda: SimpleNamespace(models=[SimpleNamespace(model=m) for m in (mod.resident or [])])
    monkeypatch.setitem(sys.modules, "ollama", mod)
    return mod


def test_pin_keep_alive_adds_default_but_respects_explicit_and_is_idempotent(fake_ollama):
    from supervisor import pin_ollama_keep_alive
    pin_ollama_keep_alive("10m")
    pin_ollama_keep_alive("99m")                       # second call must not wrap again
    import ollama
    ollama.chat(model="m")
    ollama.chat(model="m", keep_alive="1h")
    assert [c["keep_alive"] for c in fake_ollama.calls] == ["10m", "1h"]


def test_warm_models_loads_each_once_and_reports_ok_when_all_resident(fake_ollama):
    from supervisor import warm_models
    fake_ollama.resident = ["qwen3.5:4b", "phi3.5:3.8b"]
    notes = warm_models(["qwen3.5:4b", "qwen3.5:4b", "phi3.5:3.8b"], keep_alive="5m")
    gens = [c[1] for c in fake_ollama.calls if isinstance(c, tuple)]
    assert [g["model"] for g in gens] == ["qwen3.5:4b", "phi3.5:3.8b"] and all(g["keep_alive"] == "5m" for g in gens)
    assert len(notes) == 1 and notes[0].startswith("models resident")


def test_warm_models_explains_eviction_when_a_model_did_not_stay_loaded(fake_ollama):
    from supervisor import warm_models
    fake_ollama.resident = ["phi3.5:3.8b"]
    notes = warm_models(["qwen3.5:4b", "phi3.5:3.8b"])
    assert len(notes) == 1 and "qwen3.5:4b" in notes[0] and "OLLAMA_MAX_LOADED_MODELS=2" in notes[0]


def test_warm_models_never_raises(fake_ollama):
    from supervisor import warm_models

    def boom(**kw): raise RuntimeError("no server")
    fake_ollama.generate = boom
    fake_ollama.ps = boom
    notes = warm_models(["qwen3.5:4b"])
    assert any("could not load" in n for n in notes) and any("could not check" in n for n in notes)


def test_default_llm_and_auditor_pass_keep_alive(fake_ollama):
    from supervisor import KEEP_ALIVE, default_llm
    default_llm("m")([{"role": "user", "content": "hi"}])
    assert fake_ollama.calls[-1]["keep_alive"] == KEEP_ALIVE
    sys.path.insert(0, str(_ROOT / "src" / "auditor"))
    from auditor import AuditorAgent
    got = {}
    client = SimpleNamespace(chat=lambda **kw: got.update(kw) or {"message": {"content": json.dumps(
        {"is_sufficient": True, "chunk_evaluations": [], "recommended_search_query": None})}})
    AuditorAgent(client=client, keep_alive="7m").evaluate("q", "q", [{"id": "a", "text": "t", "headings": ""}])
    assert got["keep_alive"] == "7m" and got["model"] == "phi3.5:3.8b"