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

from src.decomposer.decomposer import (                      # the very classes resolve() returns, whatever path decomposer imports them by
    CITEntities, Clarify, Decompose, IndividualTaxEntities, Intent, LLMOutput, PlannedQuery, PropertyTaxEntities,
    Refuse, SearchFilter, TaxDomain, resolve,
)
SubQueryDraft = sys.modules[LLMOutput.__module__].SubQueryDraft      # not re-exported by decomposer
from supervisor import Auditor, Chunk, RetrievalAgent, RetrievalAudit, Supervisor, agent_judge, excerpt
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
    def run(self, plan, queries=None, exclude=(), question=None):
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


def test_retrieval_agent_rescores_against_the_question_before_ordering_and_capping():
    class Reranking(FakeRetriever):
        def _rerank(self, query, ids):
            self.reranked = (query, list(ids))
            return {"web:1:0": .95, "web:2:0": .1, "web:3:0": .6}
    r = Reranking({"a": [H("web:1:0", .3), H("web:2:0", .9)], "b": [H("web:3:0", .8)]})
    p = plan(); p.queries = planned("a", "b")
    out = RetrievalAgent(r, max_chunks=2).run(p, question="the whole question")
    assert r.reranked == ("the whole question", ["web:1:0", "web:2:0", "web:3:0"])
    # each chunk keeps the better of its question score and its query score: 2 scored .9 for query "a", 3 scored .8 for "b"
    assert [(c.id, c.metadata["score"]) for c in out] == [("web:1:0", .95), ("web:2:0", .9)]
    assert [c.id for c in RetrievalAgent(r).run(p)] == ["web:2:0", "web:3:0", "web:1:0"]   # no question: query scores


def test_retrieval_agent_caps_batch_and_passes_min_score():
    r = FakeRetriever({"a": [H(f"web:{i}:0", 1 - i / 20) for i in range(1, 6)]})
    p = plan(); p.queries = planned("a")
    assert len(RetrievalAgent(r, max_chunks=3).run(p)) == 3
    assert len(RetrievalAgent(r, min_score=0.92).run(p)) == 1 and r.calls[-1][2] == 0.92


# ---------------------------------------------------------------- adapter: AuditorAgent -> Auditor
def fake_agent(evals, sufficient=False, recommended=None, missing=None, parts=None):
    """evals: (key, status, reason); an int key is the chunk number (current auditor), a str key a chunk id (older).
    parts: (part text, covered_by) pairs for the auditor's question_parts; None = an older auditor without them."""
    calls, kwargs = [], []
    def evaluate(original_query, sub_prompt, chunks, **kw):
        calls.append((original_query, sub_prompt, chunks))
        kwargs.append(kw)
        extra = {} if parts is None else {"question_parts": [SimpleNamespace(part=p, covered_by=c) for p, c in parts]}
        return SimpleNamespace(is_sufficient=sufficient, recommended_search_query=recommended, missing=missing,
                               chunk_evaluations=[SimpleNamespace(chunk=k, status=s, reason=r) if isinstance(k, int)
                                                  else SimpleNamespace(chunk_id=k, status=s, reason=r)
                                                  for k, s, r in evals], **extra)
    return SimpleNamespace(evaluate=evaluate, calls=calls, kwargs=kwargs)


def test_agent_judge_maps_chunk_numbers_to_verdicts_and_passes_chunk_fields():
    agent = fake_agent([(1, "approved", "has rate"), (2, "rejected", "off topic")],
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


def test_agent_judge_keeps_chunks_the_model_skipped_or_numbered_out_of_range():
    agent = fake_agent([(1, "rejected", "no"), (7, "rejected", "no such chunk")])
    audit = Auditor(judge=agent_judge(agent)).audit("q", [C(1), C(2)])
    assert audit.relevant == [False, True] and audit.reasons == ["no", "not evaluated by auditor"]


def test_agent_judge_still_understands_an_auditor_that_returns_chunk_ids():
    agent = fake_agent([("web:2:0", "approved", None), ("web:1:0", "rejected", "GST")])
    audit = Auditor(judge=agent_judge(agent)).audit("q", [C(1), C(2)])
    assert audit.relevant == [False, True] and audit.reasons == ["GST", ""]


def test_agent_judge_first_verdict_wins_when_the_model_repeats_a_number():
    agent = fake_agent([(1, "approved", None), (1, "rejected", "changed its mind")])
    assert Auditor(judge=agent_judge(agent)).audit("q", [C(1)]).relevant == [True]


def test_auditor_passes_tried_queries_and_year_and_reports_what_is_missing():
    agent = fake_agent([(1, "approved", None)], missing="shareholder count limit", recommended="SUTE 20 shareholders")
    audit = Auditor(judge=agent_judge(agent)).audit("q", [C(1)], tried=["sute conditions"], year=2026)
    assert agent.kwargs == [{"tried_queries": ["sute conditions"], "year": 2026}]
    assert audit.missing == "shareholder count limit" and audit.reasons == [""]


def test_sufficient_only_when_every_question_part_names_an_approved_chunk():
    chunks = [C(1), C(2)]
    ok = fake_agent([(1, "approved", None), (2, "approved", None)], sufficient=True,
                    parts=[("tax rate", 1), ("filing deadline", 2)])
    assert Auditor(judge=agent_judge(ok)).audit("q", chunks).sufficient is True
    # the model says "sufficient", but the second part names no chunk: the check overrides it and names the gap
    gap = fake_agent([(1, "approved", None), (2, "approved", None)], sufficient=True,
                     parts=[("tax rate", 1), ("filing deadline", None)])
    audit = Auditor(judge=agent_judge(gap)).audit("q", chunks)
    assert audit.sufficient is False and audit.missing == "filing deadline"
    # a part "covered" by a rejected chunk is not covered
    rejected = fake_agent([(1, "approved", None), (2, "rejected", "off topic")], sufficient=True,
                          parts=[("tax rate", 1), ("filing deadline", 2)])
    assert Auditor(judge=agent_judge(rejected)).audit("q", chunks).sufficient is False
    # 0 means an earlier-approved topic, which only exists when kept_context is given
    earlier = fake_agent([(1, "approved", None)], sufficient=True, parts=[("tax rate", 1), ("deadline", 0)])
    assert Auditor(judge=agent_judge(earlier)).audit("q", [C(1)], "- Deadlines").sufficient is True
    assert Auditor(judge=agent_judge(earlier)).audit("q", [C(1)], "").sufficient is False
    # an auditor that returns no parts keeps its own verdict
    assert Auditor(judge=agent_judge(fake_agent([(1, "approved", None)], sufficient=True))).audit("q", [C(1)]).sufficient


def test_three_argument_judges_still_work_without_tried_or_year():
    seen = []
    judge = lambda q, chunks, kept: seen.append(kept) or {"chunks": [], "suggested_queries": []}
    assert Auditor(judge=judge).audit("q", [C(1)], "k", tried=["x"], year=2026).relevant == [True] and seen == ["k"]


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
    sup = Supervisor(store, RetrievalAgent(retriever), Auditor(judge=agent_judge(agent)), llm=llm, intake=intake,
                     audit_keep_top=0)                                    # this test is about the plain cutoff

    out = sup.handle_turn("u1", [{"role": "user", "content": "what is the rate?"}], TODAY)

    # the auditor said the approved chunk is sufficient, so no decide call was needed
    assert out.status == "answered" and out.stop_reason == "auditor_sufficient" and llm.decide_calls == []
    assert [c[0] for c in retriever.calls] == ["corporate income tax rate"]
    assert all(q.filter.year == 2026 and q.filter.domain == TaxDomain.CIT for q in out.plan.queries)
    assert "17%" in llm.answered_docs and "Unrelated GST" not in llm.answered_docs
    assert out.sources == ["https://iras/web:1:0"]
    assert agent.calls[0][0].startswith("what is the rate?") and "year 2026" in agent.calls[0][0]
    assert [c["id"] for c in agent.calls[0][2]] == ["web:1:0"]       # the 0.4 chunk never reached the auditor


def test_real_decomposer_refuses_and_clarifies_through_the_supervisor(store):
    sup, llm, _ = make(store)
    refuse = resolve(LLMOutput(domain=TaxDomain.OUT_OF_SCOPE, intent=Intent.OUT_OF_SCOPE), "q", TODAY)
    assert sup.supervise("u1", refuse, "q").status == "refused"
    clar = resolve(LLMOutput(domain=TaxDomain.CIT, intent=Intent.COMPUTATION, cit=CITEntities(ya=2026)),
                   "Compute my tax for YA 2026", TODAY)
    out = sup.supervise("u1", clar, "Compute my tax for YA 2026")
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


def test_decompose_first_search_includes_the_question_as_written(store):
    llm = ScriptedLLM([D("decompose")])
    ret = FakeRetrieval((DOC,))
    sup = Supervisor(store, ret, Auditor(judge=judge_script(({1}, []))), llm=llm,
                     planner=lambda q, known, missing, tried, n: ["sub a", "sub b", "sub c"])
    p = plan(); p.planned = False                              # as resolve() builds it when the intake wrote no queries
    sup.supervise("u1", p, "what is the rate?")
    assert ret.calls[0][0] == ["rate", "sub a", "sub b"]       # the question first, then max_queries sub-queries


def test_follow_up_searches_are_capped_too(store):
    sup, _, ret = make(store, [D("search", queries=["a", "b", "c"]), D("answer")], batches=([C(1)], [C(2)]),
                       judge=judge_script(({1}, []), ({1}, [])))
    sup.max_queries = 2
    sup.supervise("u1", plan(), "q")
    assert ret.calls[1][0] == ["a", "b"]


def test_default_caps_are_small():
    from supervisor import AUDIT_MIN_SCORE, DEFAULT_CANDIDATES, DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES
    assert (DEFAULT_K, DEFAULT_MAX_CHUNKS, DEFAULT_MAX_QUERIES, DEFAULT_CANDIDATES) == (3, 6, 2, 6)
    assert AUDIT_MIN_SCORE == 0.5 and Supervisor(None, None, None, llm=lambda m, s=None: "").audit_min_score == 0.5
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
    fake_ollama.resident = ["qwen3.5:4b", "phi3.5:8b"]
    notes = warm_models(["qwen3.5:4b", "qwen3.5:4b", "phi3.5:8b"], keep_alive="5m")
    gens = [c[1] for c in fake_ollama.calls if isinstance(c, tuple)]
    assert [g["model"] for g in gens] == ["qwen3.5:4b", "phi3.5:8b"] and all(g["keep_alive"] == "5m" for g in gens)
    assert len(notes) == 1 and notes[0].startswith("models resident")


def test_warm_models_explains_eviction_when_a_model_did_not_stay_loaded(fake_ollama):
    from supervisor import warm_models
    fake_ollama.resident = ["phi3.5:3.8b"]
    notes = warm_models(["qwen3.5:4b", "phi3.5:3.8b"])
    assert len(notes) == 1 and "qwen3.5:4b" in notes[0] and "OLLAMA_MAX_LOADED_MODELS=1" in notes[0]


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
    assert got["keep_alive"] == "7m" and got["model"] == "phi3.5:3.8b" and "think" not in got


def test_auditor_sends_think_only_when_asked(fake_ollama):
    sys.path.insert(0, str(_ROOT / "src" / "auditor"))
    from auditor import AuditorAgent
    got = []
    reply = {"message": {"content": json.dumps({"is_sufficient": True, "chunk_evaluations": [], "recommended_search_query": None})}}
    client = SimpleNamespace(chat=lambda **kw: got.append(kw) or reply)
    chunk = [{"id": "a", "text": "t", "headings": ""}]
    AuditorAgent(client=client).evaluate("q", "q", chunk)
    AuditorAgent(client=client, think=False).evaluate("q", "q", chunk)
    assert "think" not in got[0] and got[1]["think"] is False


def test_judge_disables_thinking_for_qwen_auditors_only(monkeypatch):
    import auditor as auditor_mod
    created = []

    class Agent:
        def __init__(self, **kw): created.append(kw)
        def evaluate(self, **kw): return SimpleNamespace(is_sufficient=True, recommended_search_query=None, chunk_evaluations=[])
    monkeypatch.setattr(auditor_mod, "AuditorAgent", Agent)
    monkeypatch.setitem(sys.modules, "src.auditor.auditor", auditor_mod)
    for model in ("qwen3.5:4b", "phi3.5:3.8b"):
        agent_judge(model=model)("q", [C(1)], "")
    assert [c["think"] for c in created] == [False, None]


def test_warm_models_uses_the_same_context_size_as_real_calls_and_lists_sizes(fake_ollama):
    from supervisor import NUM_CTX, warm_models
    fake_ollama.resident = ["phi3.5:3.8b"]
    fake_ollama.ps = lambda: SimpleNamespace(models=[SimpleNamespace(model="phi3.5:3.8b", size=2_400_000_000)])
    notes = warm_models(["qwen3.5:4b", "phi3.5:3.8b"])
    gens = [c[1] for c in fake_ollama.calls if isinstance(c, tuple)]
    assert all(g["options"] == {"num_ctx": NUM_CTX} for g in gens)
    assert "phi3.5:3.8b (2.4 GB)" in notes[0] and "AUDITOR_MODEL=qwen3.5:4b" in notes[0]


def test_supervisor_and_store_use_exactly_the_classes_decomposer_returns():
    import decomposer, supervisor
    import src.retrieval.user_store as user_store
    assert supervisor.Decompose is decomposer.Decompose and supervisor.Clarify is decomposer.Clarify
    assert supervisor.PlannedQuery is decomposer.PlannedQuery and user_store.CITEntities is decomposer.CITEntities


def test_follow_up_queries_use_the_plans_own_classes_even_if_schemas_were_loaded_twice(store):
    import importlib.util
    spec = importlib.util.spec_from_file_location("schemas_copy", sys.modules[Decompose.__module__].__file__)
    alt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alt)                                        # same file, new module: new, "different" classes
    assert alt.SearchFilter is not SearchFilter
    p = alt.Decompose(domain=alt.TaxDomain.CIT, intent=alt.Intent.LOOKUP, entities=alt.CITEntities(), tax_year=2026,
                      queries=[alt.PlannedQuery(text="rate", filter=alt.SearchFilter(domain=alt.TaxDomain.CIT, year=2026))])
    sup, _, ret = make(store, [D("search", queries=["more"]), D("answer")], batches=([C(1)], [C(2)]),
                       judge=judge_script(({1}, []), ({1}, [])))
    assert sup.supervise("u1", p, "q").status == "answered" and ret.calls[1][0] == ["more"]


def test_trace_shows_why_the_auditor_was_unavailable(store):
    def broken(*a): raise RuntimeError("model 'phi3.5:3.8b' not found")
    sup, _, _ = make(store, [D("answer")], judge=broken)
    out = sup.supervise("u1", plan(), "q")
    assert any("auditor unavailable, kept unjudged: model 'phi3.5:3.8b' not found" in t for t in out.trace)


def test_trace_lists_each_chunk_verdict_with_score_and_reason(store):
    good = Chunk(id="web:1:0", content="a", metadata={"source": "https://iras/1", "headings": "PTE > Rates", "score": 0.91})
    bad = Chunk(id="web:2:0", content="b", metadata={"source": "https://iras/2", "headings": "GST", "score": 0.12})
    judge = lambda q, chunks, kept: {"chunks": [{"index": 1, "relevant": True, "reason": "gives the PTE tiers"},
                                                {"index": 2, "relevant": False, "reason": "about GST, not CIT"}],
                                     "suggested_queries": []}
    sup, _, _ = make(store, [D("answer")], batches=([good, bad],), judge=judge)
    sup.audit_min_score = 0                                          # audit everything, even the 0.12 chunk
    out = sup.supervise("u1", plan(), "q")
    assert "    approved [0.91] PTE > Rates: gives the PTE tiers" in out.trace
    assert "    REJECTED [0.12] GST: about GST, not CIT" in out.trace


def test_trace_skips_per_chunk_lines_when_auditor_unavailable(store):
    def broken(*a): raise RuntimeError("down")
    sup, _, _ = make(store, [D("answer")], judge=broken)
    out = sup.supervise("u1", plan(), "q")
    assert not any("REJECTED" in t or t.lstrip().startswith("approved") for t in out.trace)


def test_loop_stops_after_two_searches_in_a_row_that_approve_nothing(store):
    sup, llm, ret = make(store, [D("search", queries=["x"]), D("search", queries=["y"]), D("search", queries=["z"])],
                         batches=([C(1)], [C(2)], [C(3)]),
                         judge=judge_script((set(), []), (set(), []), (set(), [])), max_steps=6)
    out = sup.supervise("u1", plan(), "q")
    assert out.stop_reason == "no_progress" and out.status == "no_answer" and len(ret.calls) == 2


def test_a_dry_round_does_not_stop_the_loop_once_something_is_vetted(store):
    sup, _, ret = make(store, [D("search", queries=["x"]), D("search", queries=["y"]), D("answer")],
                       batches=([C(1)], [C(2)], [C(3)]),
                       judge=judge_script(({1}, []), (set(), []), (set(), [])), max_steps=6)
    out = sup.supervise("u1", plan(), "q")
    assert out.status == "answered" and len(ret.calls) == 3


def test_auditor_schema_puts_verdicts_before_sufficiency_and_caps_reason_length():
    sys.path.insert(0, str(_ROOT / "src" / "auditor"))
    from auditor import AUDITOR_PROMPT, AuditorResponse, ChunkEvaluation
    props = list(AuditorResponse.model_json_schema()["properties"])
    assert (props.index("chunk_evaluations") < props.index("is_sufficient") < props.index("missing")
            < props.index("recommended_search_query"))
    evaluation = AuditorResponse.model_json_schema()["$defs"]["ChunkEvaluation"]["properties"]
    assert list(evaluation) == ["chunk", "status", "reason"] and evaluation["chunk"]["type"] == "integer"
    assert evaluation["reason"]["anyOf"][0]["maxLength"] == 60
    assert ChunkEvaluation(chunk=1, status="approved", reason="x" * 500).reason      # an overrun must not fail validation
    assert ChunkEvaluation(chunk=1, status="approved").reason is None              # approved chunks need no reason
    assert "insufficient" not in ChunkEvaluation.model_json_schema()["properties"]["status"]["description"]
    assert "does NOT have to answer the whole question" in AUDITOR_PROMPT and "recommended_search_query" in AUDITOR_PROMPT


# ---------------------------------------------------------------- speed guards: near-duplicate queries
def test_new_queries_drops_repeats_plurals_synonyms_and_bare_numbers_but_keeps_new_topics():
    from supervisor import _new_queries
    tried = ["start-up tax exemption shareholder requirements"]
    skipped = []
    out = _new_queries(["start-up tax exemption shareholder requirements 50%",     # only adds a number
                        "Start-Up Tax Exemption shareholder conditions",           # requirements ~ conditions
                        "start-up tax exemption shareholder requirement",          # singular
                        "start-up tax exemption investment holding company",       # a different facet
                        "late payment penalty"], tried, skipped)
    assert out == ["start-up tax exemption investment holding company", "late payment penalty"]
    assert skipped == ["start-up tax exemption shareholder requirements 50%",
                       "Start-Up Tax Exemption shareholder conditions", "start-up tax exemption shareholder requirement"]


def test_one_swapped_keyword_in_a_short_query_is_a_new_query():
    from supervisor import _new_queries
    assert _new_queries(["corporate income tax rebate"], ["corporate income tax rate"]) == ["corporate income tax rebate"]
    assert _new_queries(["tax rate"], ["corporate income tax rate"]) == []          # adds no keyword
    assert _new_queries(["a", "b", "a"], []) == ["a", "b"]                         # repeats within one request too


def test_near_duplicate_follow_ups_are_not_retrieved_and_the_loop_answers(store):
    sup, llm, ret = make(store, [D("search", queries=["Rate"]), D("answer")], batches=([C(1)],),
                         judge=judge_script(({1}, [])))
    out = sup.supervise("u1", plan(), "q")                  # "Rate" repeats the planned "rate"
    assert len(ret.calls) == 1 and out.stop_reason == "no_new_queries" and out.status == "answered"
    assert any("skipped near-duplicate queries ['Rate']" in t for t in out.trace)


def test_auditor_suggestion_is_used_when_all_model_queries_are_repeats(store):
    sup, _, ret = make(store, [D("search", queries=["rates"]), D("answer")], batches=([C(1)], [C(2)]),
                       judge=judge_script(({1}, ["late payment penalty"]), ({1}, [])))
    sup.supervise("u1", plan(), "q")
    assert ret.calls[1][0] == ["late payment penalty"]


def test_follow_up_returning_only_seen_chunks_stops_the_loop(store):
    sup, llm, ret = make(store, [D("search", queries=["other"]), D("search", queries=["more"])],
                         batches=([C(1)], [C(1)]), judge=judge_script(({1}, [])))
    out = sup.supervise("u1", plan(), "q")
    assert len(ret.calls) == 2 and len(llm.decide_calls) == 1 and out.status == "answered"
    assert any("zero new chunks" in t for t in out.trace)


# ---------------------------------------------------------------- speed guards: reranker pre-filter
def S(n, score, text="x", headings=None):
    return Chunk(id=f"web:{n}:0", content=text,
                 metadata={"source": f"https://iras/{n}", "headings": headings or f"H{n}", "score": score})


def recording_judge(*steps):
    """judge_script that also records which chunk ids each audit was given."""
    inner, seen = judge_script(*steps), []
    def judge(q, chunks, kept):
        seen.append([c.id for c in chunks])
        return inner(q, chunks, kept)
    judge.seen = seen
    return judge


def test_chunks_below_the_reranker_cutoff_skip_the_auditor_and_are_traced(store):
    judge = recording_judge(({1}, []))
    sup, _, _ = make(store, [D("answer")], batches=([S(1, .9), S(2, .45), S(3, .1)],), judge=judge)
    sup.audit_keep_top = 0
    out = sup.supervise("u1", plan(), "q")
    assert judge.seen == [["web:1:0"]]
    assert "    skipped [0.45] H2: reranker score below 0.50, not audited" in out.trace
    assert any("1/1 chunks related, 2 below reranker cutoff" in t for t in out.trace)


def test_a_search_whose_new_chunks_are_all_below_the_cutoff_makes_no_auditor_call(store):
    judge = recording_judge(({1}, []))
    sup, _, ret = make(store, [D("search", queries=["more"]), D("answer")], batches=([S(1, .9)], [S(2, .3)]),
                       judge=judge)
    sup.audit_keep_top = 0
    out = sup.supervise("u1", plan(), "q")
    assert len(ret.calls) == 2 and judge.seen == [["web:1:0"]] and len(out.audits) == 1
    assert any("1 chunk(s) below reranker cutoff 0.50, none audited" in t for t in out.trace)


def test_the_best_chunks_of_a_search_are_audited_whatever_their_score(store):
    judge = recording_judge(({1}, []))
    sup, _, _ = make(store, [D("answer")], batches=([S(1, .4), S(2, .3), S(3, .2), S(4, .1), S(5, .05)],), judge=judge)
    out = sup.supervise("u1", plan(), "q")
    assert judge.seen == [["web:1:0", "web:2:0", "web:3:0"]] and out.status == "answered"
    assert "    skipped [0.10] H4: reranker score below 0.50, not audited" in out.trace


def test_high_scoring_chunks_beyond_the_top_few_are_audited_and_keep_top_zero_restores_the_plain_cutoff(store):
    chunks = [S(1, .9), S(2, .8), S(3, .7), S(4, .6), S(5, .1)]
    judge = recording_judge(({1}, []))
    sup, _, _ = make(store, [D("answer")], batches=(chunks,), judge=judge)
    sup.supervise("u1", plan(), "q")
    assert judge.seen == [["web:1:0", "web:2:0", "web:3:0", "web:4:0"]]
    judge = recording_judge(({1}, []))
    sup, _, _ = make(store, [D("answer")], batches=([S(1, .2), S(2, .4)],), judge=judge)
    sup.audit_keep_top = 0
    sup.supervise("u1", plan(), "q")
    assert judge.seen == []                                          # nothing reaches the auditor


def test_chunks_left_out_by_the_cutoff_are_not_blocked_from_later_searches(store):
    judge = recording_judge(({1}, ["more"]), ({1, 2}, []))
    sup, _, ret = make(store, [D("search", queries=["more"]), D("answer")],
                       batches=([S(1, .9), S(2, .8), S(3, .7), S(4, .1)], [S(4, .8)]), judge=judge)
    sup.supervise("u1", plan(), "q")
    assert ret.calls[1][1] == {"web:1:0", "web:2:0", "web:3:0"}      # web:4:0 was never audited, so not excluded


def test_cutoff_zero_audits_everything_and_unscored_chunks_are_always_audited(store):
    judge = recording_judge(({1, 2}, []))
    sup, _, _ = make(store, [D("answer")], batches=([S(1, .9), S(2, .1)],), judge=judge)
    sup.audit_min_score = 0
    sup.supervise("u1", plan(), "q")
    assert judge.seen == [["web:1:0", "web:2:0"]]
    judge = recording_judge(({1, 2}, []))
    sup, _, _ = make(store, [D("answer")], batches=([C(1), C(2)],), judge=judge)
    sup.supervise("u1", plan(), "q")
    assert judge.seen == [["web:1:0", "web:2:0"]]


# ---------------------------------------------------------------- speed guards: auditor says sufficient
def sufficient_judge(sufficient):
    return lambda q, chunks, kept: {"chunks": [{"index": i, "relevant": True, "reason": "r"}
                                               for i in range(1, len(chunks) + 1)],
                                    "suggested_queries": [], "sufficient": sufficient}


def test_sufficient_audit_answers_without_a_decide_call(store):
    sup, llm, _ = make(store, [D("search", queries=["more"])], judge=sufficient_judge(True))
    out = sup.supervise("u1", plan(), "q")
    assert out.stop_reason == "auditor_sufficient" and llm.decide_calls == [] and out.status == "answered"
    assert "COVERAGE NOTE" not in llm.answer_calls[0][1]["content"]


def test_insufficient_or_unreported_audit_still_asks_the_model(store):
    for verdict in (False, None):
        sup, llm, _ = make(store, [D("answer")], judge=sufficient_judge(verdict))
        sup.supervise("u1", plan(), "q")
        assert len(llm.decide_calls) == 1


def test_sufficient_audit_does_not_skip_a_calculator_that_has_not_run(store):
    p = plan(intent=Intent.COMPUTATION, chargeable_income=300000)
    sup, llm, _ = make(store, [D("run_tool", tool="compute_tax"), D("answer")], judge=sufficient_judge(True),
                       tools={"compute_tax": lambda plan: "Tax payable: 25,200"})
    out = sup.supervise("u1", p, "tax?")
    assert len(llm.decide_calls) == 2 and "Tax payable: 25,200" in llm.answered_docs and out.status == "answered"


# ---------------------------------------------------------------- the trace from the SUTE question, end to end
def sute_plan():
    return plan_with_queries("start-up tax exemption shareholder requirements",
                             "start-up tax exemption first three years of assessment")


def test_sute_trace_stops_after_one_audit_instead_of_three_searches(store):
    """Replays the first slow trace: step 1 approves the qualifying-conditions chunk; the model then asks for a
    near-repeat and a rephrase. Before: 3 retrievals, 2 audits. Now: 2 retrievals, 1 audit. (The auditor's
    suggestion here is new, so the decide call still happens.)"""
    first = [S(1, .99, headings="Qualifying conditions"), S(2, .96), S(3, .93), S(4, .81), S(5, .40)]
    judge = recording_judge(({1, 4}, ["Start-Up Tax Exemption shareholder ownership limit"]))
    decisions = [D("search", queries=["start-up tax exemption shareholder ownership percentage",
                                      "start-up tax exemption shareholder requirements 50%"]),
                 D("search", queries=["Start-Up Tax Exemption shareholder conditions"])]
    sup, llm, ret = make(store, decisions, batches=(first, [S(6, .48)]), judge=judge, max_steps=4)
    sup.audit_keep_top = 0
    out = sup.supervise("u1", sute_plan(), "Do we qualify for SUTE?")
    assert ret.calls[1][0] == ["start-up tax exemption shareholder ownership percentage"]
    assert len(ret.calls) == 2 and len(judge.seen) == 1 and judge.seen[0] == [f"web:{i}:0" for i in (1, 2, 3, 4)]
    assert out.stop_reason == "no_new_queries" and out.status == "answered"
    assert "Qualifying conditions" in llm.answered_docs


def test_sute_second_trace_answers_without_a_decide_call_when_the_suggestion_repeats(store):
    """Replays the second trace: the auditor's only suggestion ("... shareholder limit rules" ~ the planned
    "... shareholder requirements") is a repeat, so step 2 answers without a decide call (was 3.5s)."""
    first = [S(1, .99, headings="Qualifying conditions"), S(2, .96), S(3, .93), S(4, .81), S(5, .40)]
    judge = recording_judge(({1}, ["Start-Up Tax Exemption shareholder limit rules"]))
    sup, llm, ret = make(store, [D("search", queries=["should never be asked"])], batches=(first,), judge=judge)
    out = sup.supervise("u1", sute_plan(), "Do we qualify for SUTE?")
    assert llm.decide_calls == [] and len(ret.calls) == 1 and len(judge.seen) == 1
    assert out.stop_reason == "suggestions_repeat" and out.status == "answered"
    assert "decide" not in dict(out.timings)
    assert any("auditor's suggestions ['Start-Up Tax Exemption shareholder limit rules'] repeat searched queries"
               in t for t in out.trace)


# ---------------------------------------------------------------- warming and the harness flags
def test_retrieval_agent_warm_runs_one_search_and_never_raises():
    r = FakeRetriever({})
    assert RetrievalAgent(r).warm().startswith("retriever warm") and r.calls == [("corporate income tax rate", 1, None)]

    class Broken:
        def search(self, *a, **kw): raise RuntimeError("no model")
    assert "could not warm the retriever: no model" in RetrievalAgent(Broken()).warm()


def test_harness_defaults_and_build_supervisor_pass_the_speed_knobs(store):
    import harness
    args = harness.parse_args(["--user", "u"])
    assert (args.candidates, args.audit_min_score, args.quantize_rerank) == (6, 0.5, False)
    sup = harness.build_supervisor(store=store, retriever=FakeRetriever({}), candidates=4, audit_min_score=0.3)
    assert sup.retrieval.candidates == 4 and sup.audit_min_score == 0.3
    assert harness.build_supervisor(store=store, retriever=FakeRetriever({}), candidates=0).retrieval.candidates is None


# ---------------------------------------------------------------- speed guards: auditor's suggestions all repeat
def test_repeated_suggestions_skip_the_decide_call(store):
    sup, llm, ret = make(store, [D("search", queries=["late payment penalty"])], batches=([C(1)],),
                         judge=judge_script(({1}, ["rates"])))          # "rates" repeats the planned "rate"
    out = sup.supervise("u1", plan(), "q")
    assert llm.decide_calls == [] and len(ret.calls) == 1
    assert out.stop_reason == "suggestions_repeat" and out.status == "answered"


def test_repeated_suggestions_with_a_missing_fact_search_that_fact_before_answering(store):
    verdicts = iter([{"chunks": [{"index": 1, "relevant": True, "reason": ""}], "suggested_queries": ["rates"],
                      "sufficient": False, "missing": "late payment penalty"},
                     {"chunks": [{"index": 1, "relevant": True, "reason": ""}], "suggested_queries": [],
                      "sufficient": True, "missing": None}])
    judge = lambda question, chunks, kept: next(verdicts)
    sup, llm, ret = make(store, [], batches=([C(1)], [C(2)]), judge=judge)
    out = sup.supervise("u1", plan(), "q")
    assert [c[0] for c in ret.calls] == [["rate"], ["late payment penalty"]]      # "rates" repeated; the gap was searched
    assert llm.decide_calls == [] and out.status == "answered" and out.stop_reason == "auditor_sufficient"
    # a missing fact that was already searched cannot loop: the old stop applies
    verdicts = iter([{"chunks": [{"index": 1, "relevant": True, "reason": ""}], "suggested_queries": ["rates"],
                      "sufficient": False, "missing": "rates"}])
    sup, llm, ret = make(store, [], batches=([C(1)],), judge=lambda question, chunks, kept: next(verdicts))
    assert sup.supervise("u1", plan(), "q").stop_reason == "suggestions_repeat" and len(ret.calls) == 1


def test_a_new_suggestion_or_no_suggestion_still_asks_the_model(store):
    for sugg in (["late payment penalty"], ["rates", "late payment penalty"], []):
        sup, llm, _ = make(store, [D("answer")], batches=([C(1)],), judge=judge_script(({1}, sugg)))
        sup.supervise("u1", plan(), "q")
        assert len(llm.decide_calls) == 1, sugg


def test_repeated_suggestions_do_not_skip_when_nothing_is_vetted_or_a_calculator_is_pending(store):
    sup, llm, _ = make(store, [D("give_up")], batches=([C(1)],), judge=judge_script((set(), ["rates"])))
    assert sup.supervise("u1", plan(), "q").stop_reason == "agent_gave_up" and len(llm.decide_calls) == 1
    p = plan(intent=Intent.COMPUTATION, chargeable_income=300000)
    sup, llm, _ = make(store, [D("run_tool", tool="compute_tax"), D("answer")], batches=([C(1)],),
                       judge=judge_script(({1}, ["rates"])), tools={"compute_tax": lambda plan: "Tax payable: 25,200"})
    out = sup.supervise("u1", p, "tax?")
    assert "Tax payable: 25,200" in llm.answered_docs and out.status == "answered"


# ---------------------------------------------------------------- the auditor prompt itself
def captured_auditor_prompt(chunks, **kw):
    sys.path.insert(0, str(_ROOT / "src" / "auditor"))
    from auditor import AuditorAgent
    got = {}
    reply = {"message": {"content": json.dumps({"chunk_evaluations": [{"chunk": 1, "status": "approved"}],
                                                "is_sufficient": True})}}
    client = SimpleNamespace(chat=lambda **c: got.update(c) or reply)
    resp = AuditorAgent(client=client, think=False).evaluate("Do we qualify?", "ctx", chunks, **kw)
    return got["messages"][0]["content"], resp


def test_auditor_prompt_numbers_chunks_and_hides_their_ids():
    chunks = [{"id": "web:a5627a3f6e:1", "headings": "SUTE > Conditions", "text": "no more than 20 shareholders"},
              {"id": "web:b1:2", "headings": "Table 2", "text": "YA 2019 and before"}]
    prompt, resp = captured_auditor_prompt(chunks)
    assert "--- Chunk 1 ---" in prompt and "--- Chunk 2 ---" in prompt and "web:a5627a3f6e" not in prompt
    assert resp.chunk_evaluations[0].chunk == 1 and resp.chunk_evaluations[0].reason is None
    from auditor import AuditorAgent
    assert AuditorAgent.chunk_id_of(resp.chunk_evaluations[0], chunks) == "web:a5627a3f6e:1"
    resp.chunk_evaluations[0].chunk = 3
    assert AuditorAgent.chunk_id_of(resp.chunk_evaluations[0], chunks) is None


def test_auditor_prompt_states_the_year_the_mixed_year_rule_and_the_queries_already_searched():
    chunk = [{"id": "a", "headings": "h", "text": "t"}]
    prompt, _ = captured_auditor_prompt(chunk, tried_queries=["sute shareholder requirements"], year=2026)
    assert "### Year of Assessment asked: YA 2026" in prompt
    assert "- sute shareholder requirements" in prompt and "never suggest these again" in prompt
    assert "Mixed years" in prompt and "is APPROVED" in prompt and "EVERYTHING in it applies only to years" in prompt
    prompt, _ = captured_auditor_prompt(chunk)
    assert "### Year of Assessment asked: not stated" in prompt and "- none yet" in prompt


def test_supervisor_gives_the_auditor_its_tried_queries_and_the_plan_year(store):
    seen = []
    def judge(q, chunks, kept, tried=(), year=None):
        seen.append((list(tried), year))
        return {"chunks": [{"index": 1, "relevant": True, "reason": ""}], "suggested_queries": [],
                "sufficient": False, "missing": "the 10% individual shareholder rule"}
    sup, llm, _ = make(store, [D("answer")], judge=judge)
    out = sup.supervise("u1", plan(), "q")
    assert seen == [(["rate"], 2026)]
    assert "missing: the 10% individual shareholder rule" in llm.decide_calls[0][1]["content"]
    assert any(t.startswith("    approved ") and not t.rstrip().endswith(":") for t in out.trace)


def test_null_written_as_a_word_is_treated_as_no_value():
    sys.path.insert(0, str(_ROOT / "src" / "auditor"))
    from auditor import AuditorResponse
    r = AuditorResponse.model_validate_json(json.dumps({
        "chunk_evaluations": [{"chunk": 1, "status": "approved", "reason": "null"},
                              {"chunk": 2, "status": "rejected", "reason": "about GST"}],
        "is_sufficient": False, "missing": "None", "recommended_search_query": "null"}))
    assert [e.reason for e in r.chunk_evaluations] == [None, "about GST"]
    assert r.missing is None and r.recommended_search_query is None


def test_agent_judge_drops_null_words_from_reasons_missing_and_suggestions(store):
    agent = fake_agent([(1, "approved", "null"), (2, "rejected", "GST")], missing="null", recommended="None")
    audit = Auditor(judge=agent_judge(agent)).audit("q", [C(1), C(2)])
    assert audit.reasons == ["", "GST"] and audit.missing is None and audit.suggested_queries == []
    good = Chunk(id="web:1:0", content="a", metadata={"source": "https://iras/1", "headings": "SUTE", "score": 0.99})
    sup, _, _ = make(store, [D("answer")], batches=([good],),
                     judge=agent_judge(fake_agent([(1, "approved", "null")])))
    out = sup.supervise("u1", plan(), "q")
    assert "    approved [0.99] SUTE" in out.trace


# ---------------------------------------------------------------- excerpt: which part of a long chunk a model sees
def test_excerpt_keeps_a_short_chunk_whole():
    assert excerpt("short text", "anything", 50) == "short text"


def test_excerpt_moves_the_window_to_the_lines_that_match_the_question():
    text = "\n".join(["intro about filing deadlines"] * 20 + ["foreign worker levy rebate deduction rule"])
    out = excerpt(text, "Can my company deduct the foreign worker levy?", 120)
    assert out.startswith("intro about filing deadlines\n...\n")          # the chunk's opening line is kept
    assert "foreign worker levy rebate deduction rule" in out and not out.endswith("...")
    assert len(out) <= 120 + 8


def test_excerpt_cuts_plainly_when_the_best_line_is_already_in_view():
    text = "foreign worker levy rule\n" + "unrelated filler\n" * 30
    assert excerpt(text, "foreign worker levy", 100) == text[:100] + "..."


def test_excerpt_keeps_the_start_on_a_tie_and_marks_the_cut():
    text = "\n".join(f"line {i} unrelated" for i in range(40))
    out = excerpt(text, "start-up tax exemption", 100)
    assert out.startswith("line 0") and out.endswith("...")


def test_answer_prompt_and_audit_show_the_matching_part_of_a_long_chunk(store):
    long = Chunk(id="web:1:0", content="filler line\n" * 300 + "The 250% deduction for donations is granted.",
                 metadata={"source": "https://iras/1", "headings": "Donations", "score": 0.9})
    agent = fake_agent([(1, "approved", None)], sufficient=True)
    sup, llm, _ = make(store, [D("answer")], batches=([long],), judge=agent_judge(agent))
    sup.supervise("u1", plan(), "How much deduction do we get for donations?")
    assert "250% deduction for donations" in llm.answered_docs
    assert "250% deduction for donations" in agent.calls[0][2][0]["text"]

