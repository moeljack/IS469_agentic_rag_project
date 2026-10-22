"""Offline tests for the supervisor harness: no Ollama, no embedding models, in-memory ChromaDB."""
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]                    # repo root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src" / "inference"))           # schemas, intake_planner
sys.path.insert(0, str(Path(__file__).resolve().parent))       # supervisor

import chromadb
import pytest

from src.inference.intake_planner import resolve
from schemas import Clarify, Entities, Intent, LLMOutput, Plan, PlannedQuery, Refuse, SearchFilter, SubQuery
from supervisor import Auditor, Chunk, RetrievalAgent, RetrievalAudit, Supervisor, agent_judge
from src.retrieval.user_store import UserStore

TODAY = date(2026, 10, 20)
DOC = Chunk(id="web:1:0", content="Your company is taxed at a flat rate of 17% of its chargeable income.",
            metadata={"source": "https://www.iras.gov.sg/rate"})


def plan(**ent):
    e = Entities(**ent)
    return Plan(intent=Intent.LOOKUP, entities=e, ya=2026,
                queries=[PlannedQuery(text="rate", filter=SearchFilter(ya=2026))], tools=[])


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
    return [PlannedQuery(text=t, filter=SearchFilter(ya=2026)) for t in texts]


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
    changes = store.remember("u1", Entities(fye="2025-12-31", chargeable_income=300000, ya=2026), "q1")
    assert changes == [("fye", None, "2025-12-31")]
    assert store.profile_entities("u1").fye == "2025-12-31"
    assert store.profile_entities("u1").chargeable_income is None        # per-question fact is not a default
    assert {h["field"] for h in store.history("u1")} == {"fye", "chargeable_income", "ya"}


def test_changed_value_overwrites_but_history_keeps_old(store):
    store.remember("u1", Entities(fye="2025-12-31"), "a")
    changes = store.remember("u1", Entities(fye="2025-06-30"), "b")
    assert changes == [("fye", "2025-12-31", "2025-06-30")]
    assert [h["value"] for h in store.history("u1", "fye")] == ["2025-12-31", "2025-06-30"]


def test_users_are_isolated_and_forget_deletes(store):
    store.remember("u1", Entities(fye="2025-12-31"), "a")
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


def test_tool_runs_only_if_planned_and_registered_and_output_is_a_document(store):
    p = plan(); p.tools = ["compute_tax", "missing_tool"]
    sup, llm, _ = make(store, [D("run_tool", tool="compute_tax"), D("run_tool", tool="missing_tool"),
                               D("run_tool", tool="compute_tax"), D("answer")],
                       tools={"compute_tax": lambda plan: "Tax payable: 25,200"}, max_steps=5)
    out = sup.supervise("u1", p, "tax?")
    assert out.status == "answered" and "Tax payable: 25,200" in llm.answered_docs
    assert any("missing_tool" in t for t in out.trace)


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
    out = sup.supervise("u1", Clarify(question="FYE?", missing=["fye"], entities=Entities(chargeable_income=5)), "q")
    assert out.status == "clarify"
    assert store.pop_pending("u1").chargeable_income == 5
    store.save_pending("u1", Entities(ya=2026))
    sup.supervise("u1", Refuse(reason="out of scope"), "gst?")
    assert store.pop_pending("u1") is None


def test_user_facts_are_citable_so_answer_may_use_them(store):
    sup, _, _ = make(store, answer="For your company with FYE 2025-12-31 the rate is 17% [Doc 1] [Doc 2].")
    out = sup.supervise("u1", plan(fye="2025-12-31"), "rate for me?")
    assert out.status == "answered"


def test_tool_output_is_context_and_unregistered_tools_are_skipped(store):
    p = plan(); p.tools = ["compute_tax", "missing_tool"]
    sup, _, _ = make(store, answer="Tax payable is 25,200 [Doc 1].")
    sup.tools = {"compute_tax": lambda plan: "Tax payable: 25,200"}
    out = sup.supervise("u1", p, "tax?")
    assert out.status == "answered" and any("missing_tool" in t for t in out.trace)


# ---------------------------------------------------------------- full turn
def test_handle_turn_reuses_saved_fye_and_discloses_it(store):
    store.remember("u1", Entities(fye="2025-12-31"), "earlier")
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
            return Clarify(question="FYE?", missing=["fye"], entities=Entities(chargeable_income=300000))
        return plan(fye=prior.fye, chargeable_income=prior.chargeable_income)
    sup, _, _ = make(store, [D("answer")], intake=intake)
    first = sup.handle_turn("u1", [{"role": "user", "content": "tax on 300k?"}], TODAY)
    assert first.status == "clarify"
    # simulate the user answering; the pending income must come back as prior, with no fye yet
    store.remember("u1", Entities(fye="2025-12-31"), "31 Dec")
    second = sup.handle_turn("u1", [{"role": "user", "content": "31 Dec"}], TODAY)
    assert calls[1].chargeable_income == 300000 and second.status == "answered"


# ---------------------------------------------------------------- the real pieces wired together
def test_real_planner_feeds_retriever_adapter_and_auditor_adapter(store):
    """intake_planner.resolve -> Supervisor -> RetrievalAgent(Retriever stand-in) -> Auditor(AuditorAgent stand-in)."""
    raw = LLMOutput(intent=Intent.LOOKUP, entities=Entities(ya=2026),
                    sub_queries=[SubQuery(query="corporate income tax rate")])

    def intake(messages, today, prior):                                   # the deterministic half of intake_planner
        return resolve(raw, messages[-1]["content"], today, prior)

    retriever = FakeRetriever({
        "corporate income tax rate": [H("web:1:0", .9, "Taxed at a flat rate of 17%.")],
        "what is the rate?": [H("web:1:0", .8, "Taxed at a flat rate of 17%."), H("web:2:0", .4, "Unrelated GST text")],
    })
    agent = fake_agent([("web:1:0", "approved", "gives the rate"), ("web:2:0", "rejected", "GST")], sufficient=True)
    llm = ScriptedLLM([D("answer")], "The rate is 17% [Doc 1].")
    sup = Supervisor(store, RetrievalAgent(retriever), Auditor(judge=agent_judge(agent)), llm=llm, intake=intake)

    out = sup.handle_turn("u1", [{"role": "user", "content": "what is the rate?"}], TODAY)

    assert out.status == "answered" and out.stop_reason == "agent_answered"
    assert [c[0] for c in retriever.calls] == ["corporate income tax rate", "what is the rate?"]
    assert all(q.filter.ya == 2026 for q in out.plan.queries)
    assert "17%" in llm.answered_docs and "Unrelated GST" not in llm.answered_docs
    assert out.sources == ["https://iras/web:1:0"]
    assert agent.calls[0][0].startswith("what is the rate?") and "year of assessment 2026" in agent.calls[0][0]