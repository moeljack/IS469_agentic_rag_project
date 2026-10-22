"""Supervisor agent: a model (local qwen3.5:4b via Ollama) decides each next action and writes the final answer.

    messages -> decomposer -> Decompose | Clarify | Refuse
                                      |
        user store (ChromaDB) <-------+  profile in as `prior`; new facts out; pending state across clarifications
                                      v
      +--------------------- supervisor model picks one action per step ----------------------+
      |  search(queries)  -> RetrievalAgent (wraps retriever.Retriever: hybrid + rerank)       |
      |                      -> Auditor (wraps auditor.AuditorAgent) judges each chunk          |
      |                      -> approved chunks are kept; auditor's suggested query is reported |
      |  run_tool(name)   -> e.g. compute_tax, output becomes a citable document               |
      |  ask_user(q)      -> status "clarify" (entities saved so the next turn resumes)        |
      |  answer           -> supervisor writes the final answer from vetted chunks             |
      |  give_up          -> status "no_answer"                                                |
      +-----------------------------------------------------------------------------------------+

Integration layer (this file): the real components have different shapes from what the loop needs, so two small
adapters live here instead of changing them:
  * RetrievalAgent  - .run(plan, queries=None, exclude=()) -> list[Chunk], built on Retriever.search()
  * Auditor         - .audit(question, chunks, kept_context) -> RetrievalAudit, built on AuditorAgent.evaluate()
                      (via agent_judge). Any failure fails open: chunks are kept unjudged and the trace says so.

Guard rails in code (a 4B model is not trusted with the loop itself):
  * step 1 is always `search` with the planner's queries
  * a budget of `max_steps` actions, after which the supervisor must answer with what it has
  * `answer` with no vetted chunks and no tool output becomes give_up (never an answer from nothing)
  * invalid actions (unknown tool, repeated queries, empty ask_user) are logged and cost a step
  * unparseable model output falls back to: answer if anything is vetted, else give_up

All model calls go through one injectable `llm(messages, schema=None) -> str`, and the auditor takes an injectable
`judge`, so tests need no Ollama. Two entry points:
  handle_turn(user_id, messages)  full harness: loads the profile, runs intake, supervises the result
  supervise(user_id, result, question)  takes an already-computed decomposer Result
"""
import functools
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).resolve().parents[2]                    # repo root, so `src.<package>.<module>` imports work
sys.path.insert(0, str(_ROOT))
for _d in ("inference", "decomposer"):                         # decomposer.py + decomposer_general_schemas.py import each other
    sys.path.insert(0, str(_ROOT / "src" / _d))                # as top-level modules; put the folder they live in here

from src.decomposer.decomposer_general_schemas import (
    CITEntities, Clarify, Decompose, DomainEntities, Intent, PlannedQuery, Refuse, SearchFilter,
)
from src.retrieval.user_store import PERSISTENT_FIELDS

MODEL = os.environ.get("SUPERVISOR_MODEL", os.environ.get("INTAKE_MODEL", "qwen3.5:4b"))
AUDITOR_MODEL = os.environ.get("AUDITOR_MODEL", "phi3.5:3.8b")   # the auditor runs its own local model, not the supervisor's
NUM_CTX = 8192
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE_CLIENT", "30m")   # how long Ollama keeps our models in memory after a call

# Speed knobs: every chunk costs a reranker pass, auditor prompt tokens and answer prompt tokens, and every planned
# query costs a full retrieval (embedding + BM25 + rerank). These defaults trade a little recall for a lot of latency;
# harness.py exposes them as flags (--k, --max-chunks, --max-queries, --candidates).
DEFAULT_K = 3                       # chunks kept per query after reranking
DEFAULT_MAX_CHUNKS = 6              # chunks handed to the auditor per search
DEFAULT_MAX_QUERIES = 2             # planned / follow-up queries per search (the decomposer may plan up to 4)
DOC_CHARS = 1200                    # per-chunk text shown to the model when answering
AUDIT_CHARS = 800                   # per-chunk text shown to the auditor
MAX_DOCS = 6                        # vetted chunks put in the answer prompt

NO_CONTEXT = "The IRAS sources I have don't cover this. Please check iras.gov.sg or rephrase your question."

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "run_tool", "ask_user", "answer", "give_up"]},
        "queries": {"type": "array", "items": {"type": "string"}},
        "tool": {"type": "string"},
        "question": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

DECIDE_SYSTEM = """You supervise a Singapore corporate income tax assistant. Each turn you choose ONE next action.
Actions:
- search: retrieve IRAS documents. Give 1-3 keyword queries in "queries" (official IRAS terms, different from queries
  already tried). An auditor then checks which retrieved chunks are related to the question and may suggest queries.
- run_tool: run one of the AVAILABLE TOOLS (give its name in "tool") when the question needs a calculation.
- ask_user: ONLY if an essential fact is missing that no document can supply. Put the question in "question".
- answer: you have enough related documents (or tool output) to answer the question.
- give_up: the documents cannot answer this and further searching will not help.
Rules: search again only when the covered topics miss part of the question; use the auditor's suggestions when given.
Never invent rates, caps or numbers. Reply with JSON: {"action": ..., "queries": [...], "tool": "", "question": "", "reason": "..."}"""

ANSWER_SYSTEM = """You are a Singapore corporate income tax assistant. Answer the QUESTION using ONLY the DOCUMENTS.
- Cite the document for every factual claim as [Doc n].
- Facts under USER FACTS were given by the user: you may rely on them, but do not cite them.
- Never state a rate, cap, threshold, date or amount that is not in the documents or tool output.
- If COVERAGE NOTE is present, say plainly which part of the question the documents do not cover.
- If the documents do not contain the answer, reply exactly: INSUFFICIENT_CONTEXT: The provided documents do not contain this information.
Be concise and direct."""


# ---------------------------------------------------------------- data shared by the adapters and the loop
@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalAudit:
    """The auditor's verdict on one batch. `relevant[i]` is the verdict for batch[i]."""
    relevant: list[bool]
    reasons: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    sufficient: Optional[bool] = None          # auditor's "the approved chunks fully answer it"; None = not reported
    unavailable: bool = False                  # auditor failed: chunks were kept unjudged

    @property
    def all_relevant(self) -> bool:
        return all(self.relevant) and self.sufficient is not False


# ---------------------------------------------------------------- adapter 1: retriever.Retriever -> RetrievalAgent
class RetrievalAgent:
    """Runs the plan's queries through the real Retriever and returns Chunks the supervisor can vet.

    Retriever.search() takes one query string and knows nothing about plans, so this adapter:
      * runs each PlannedQuery (or the queries the supervisor chose) separately,
      * drops chunks already judged (`exclude`) and duplicates across queries,
      * orders what is left by reranker score (0-1, comparable across queries) and caps the batch,
        so the auditor and the 8k-token model are never handed a wall of chunks.
    The plan's SearchFilter (year / doc type) is not applied: the collection carries no such metadata yet.
    """

    def __init__(self, retriever, k: int = DEFAULT_K, min_score: Optional[float] = None,
                 max_chunks: int = DEFAULT_MAX_CHUNKS, candidates: Optional[int] = None):
        """candidates: size of the pool the reranker scores per query (None = the retriever's CANDIDATES setting).
        The reranker is the heaviest step on CPU, so this is the biggest lever on retrieval time."""
        self.retriever, self.k, self.min_score, self.max_chunks = retriever, k, min_score, max_chunks
        self.candidates = candidates
        self.last_notes: list[str] = []

    def run(self, plan: Decompose, queries=None, exclude=()) -> list[Chunk]:
        self.last_notes = []
        chosen = queries if queries is not None else plan.queries
        seen, found = set(exclude), {}
        for q in chosen:
            extra = {} if self.candidates is None else {"candidates": self.candidates}
            hits = self.retriever.search(q.text, k=self.k, min_score=self.min_score, **extra)
            new = 0
            for h in hits:
                if h["id"] in seen:
                    continue
                seen.add(h["id"])
                found[h["id"]] = h
                new += 1
            self.last_notes.append(f"retrieved {len(hits)} chunk(s), {new} new, for {q.text!r}")
        ranked = sorted(found.values(), key=lambda h: h.get("score") or 0.0, reverse=True)[: self.max_chunks]
        return [Chunk(id=h["id"], content=h["text"],
                      metadata={"source": h.get("source", ""), "headings": h.get("headings", ""),
                                "pages": h.get("pages", ""), "source_type": h.get("source_type", ""),
                                "title": h.get("title", ""), "score": h.get("score")})
                for h in ranked]


# ---------------------------------------------------------------- adapter 2: auditor.AuditorAgent -> Auditor
def agent_judge(agent=None, model: str = AUDITOR_MODEL) -> Callable:
    """Build a judge(question, chunks, kept_context) -> dict on top of AuditorAgent.evaluate().

    `agent` is anything with .evaluate(original_query, sub_prompt, chunks) returning an object with
    is_sufficient, recommended_search_query and chunk_evaluations[chunk_id, status, reason]. When omitted, a real
    AuditorAgent is created on first use against the local Ollama server (so importing this module needs
    no running Ollama).
    """
    holder = {"agent": agent}

    def judge(question: str, chunks: list, kept_context: str) -> dict:
        if holder["agent"] is None:
            from src.auditor.auditor import AuditorAgent
            holder["agent"] = AuditorAgent(model=model, keep_alive=KEEP_ALIVE)
        sub_prompt = question
        if kept_context.strip():
            sub_prompt += f"\n(Topics already covered by approved chunks:\n{kept_context})"
        payload = [{"id": c.id, "headings": c.metadata.get("headings", ""), "text": c.content[:AUDIT_CHARS]}
                   for c in chunks]
        resp = holder["agent"].evaluate(original_query=question, sub_prompt=sub_prompt, chunks=payload)
        by_id = {str(e.chunk_id).strip(): e for e in resp.chunk_evaluations}
        rows = []
        for i, c in enumerate(chunks, 1):
            e = by_id.get(c.id)
            if e is None:       # the small model skipped this chunk: keep it rather than silently lose evidence
                rows.append({"index": i, "relevant": True, "reason": "not evaluated by auditor"})
            else:
                rows.append({"index": i, "relevant": e.status == "approved", "reason": e.reason})
        suggested = [resp.recommended_search_query] if resp.recommended_search_query else []
        return {"chunks": rows, "suggested_queries": suggested, "sufficient": resp.is_sufficient}

    return judge


class Auditor:
    """Judges which retrieved chunks are related to the question. Fails open (keeps everything) if the judge breaks."""

    def __init__(self, judge: Optional[Callable] = None, model: str = AUDITOR_MODEL):
        self.judge = judge or agent_judge(model=model)

    def audit(self, question: str, chunks: list, kept_context: str = "") -> RetrievalAudit:
        n = len(chunks)
        try:
            verdict = self.judge(question, chunks, kept_context)
            rows = {int(r["index"]): r for r in verdict.get("chunks", [])}
            relevant = [bool(rows[i]["relevant"]) if i in rows else True for i in range(1, n + 1)]
            reasons = [str(rows[i].get("reason", "")) if i in rows else "" for i in range(1, n + 1)]
            suggested = [re.sub(r"\s+", " ", str(q)).strip() for q in verdict.get("suggested_queries", [])]
            return RetrievalAudit(relevant=relevant, reasons=reasons, suggested_queries=[q for q in suggested if q],
                                  sufficient=verdict.get("sufficient"))
        except Exception as e:
            return RetrievalAudit(relevant=[True] * n, reasons=[f"auditor unavailable: {e}"] * n, unavailable=True)


@dataclass
class SupervisorResult:
    status: str                         # answered | clarify | refused | no_answer
    message: str                        # what to show the user
    plan: Optional[Decompose] = None
    draft: str = ""                     # raw answer from the model
    audits: list[RetrievalAudit] = field(default_factory=list)   # one per search action
    stop_reason: str = ""               # agent_answered | agent_gave_up | max_steps | no_new_queries | no_context | ...
    sources: list[str] = field(default_factory=list)
    profile_changes: list[tuple] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    timings: list[tuple] = field(default_factory=list)   # [(stage, seconds)], one entry per call


def _timed(sink: list, name: str, fn: Callable, *args, **kw):
    """Call fn and record how long it took under `name`, even if it raises."""
    t0 = time.perf_counter()
    try:
        return fn(*args, **kw)
    finally:
        sink.append((name, time.perf_counter() - t0))


def timing_line(timings: list[tuple]) -> str:
    """'timing: intake 2.1s | retrieval 3.4s (x2) | audit 5.2s (x2) | answer 4.0s | other 0.3s | total 15.0s'"""
    agg: dict = {}
    for name, secs in timings:
        total, n = agg.get(name, (0.0, 0))
        agg[name] = (total + secs, n + 1)
    parts = [f"{n} {t:.1f}s" + (f" (x{c})" if c > 1 else "") for n, (t, c) in agg.items() if n != "total"]
    if "total" in agg:
        other = agg["total"][0] - sum(t for n, (t, _) in agg.items() if n != "total")
        if other >= 0.05:
            parts.append(f"other {other:.1f}s")                  # profile store, model loading, glue
        parts.append(f"total {agg['total'][0]:.1f}s")
    return "timing: " + " | ".join(parts)


@dataclass
class _Run:
    """Working memory of one supervised question."""
    vetted: dict = field(default_factory=dict)          # chunk id -> Chunk, only auditor-approved
    judged: set = field(default_factory=set)
    audits: list = field(default_factory=list)
    tool_out: dict = field(default_factory=dict)        # tool name -> text
    timings: list = field(default_factory=list)         # [(stage, seconds)]
    tried: list = field(default_factory=list)           # queries already searched
    log: list = field(default_factory=list)             # one line per action, shown back to the model
    suggestions: list = field(default_factory=list)     # auditor's latest suggested queries


# ---------------------------------------------------------------- Ollama runtime: keep the models loaded
def pin_ollama_keep_alive(keep_alive: str = KEEP_ALIVE) -> None:
    """Make every `ollama.chat(...)` in this process ask Ollama to keep its model loaded for `keep_alive`.

    Ollama unloads a model 5 minutes after its last call unless the call says otherwise; the decomposer's call
    doesn't, so without this its model can be evicted between turns. An explicit keep_alive on a call still wins.
    """
    import ollama
    if getattr(ollama.chat, "_keep_alive_pinned", False):
        return
    original = ollama.chat

    @functools.wraps(original)
    def chat(*args, **kwargs):
        kwargs.setdefault("keep_alive", keep_alive)
        return original(*args, **kwargs)

    chat._keep_alive_pinned = True
    ollama.chat = chat


def warm_models(models: list[str], keep_alive: str = KEEP_ALIVE) -> list[str]:
    """Load each Ollama model now (empty prompt) so the first question doesn't pay for it, then check they all
    stayed resident. Returns human-readable notes: problems, or an 'ok' line. Never raises."""
    import ollama
    models = list(dict.fromkeys(models))
    notes = []
    for m in models:
        try:
            ollama.generate(model=m, prompt="", keep_alive=keep_alive)
        except Exception as e:
            notes.append(f"could not load {m}: {e}")
    try:
        resp = ollama.ps()
        items = getattr(resp, "models", None)
        if items is None:
            items = resp.get("models", [])
        loaded = set()
        for it in items:
            name = getattr(it, "model", None) or (it.get("model") if hasattr(it, "get") else None)
            if name:
                loaded.add(name)
    except Exception as e:
        return notes + [f"could not check which models are loaded: {e}"]
    missing = [m for m in models if m not in loaded and not any(n.startswith(m) for n in loaded)]
    if missing:
        notes.append(f"{missing} not resident after loading ({sorted(loaded) or 'nothing loaded'}). Ollama is evicting "
                     "models to fit memory, so each turn will reload them. Start the Ollama server with "
                     "OLLAMA_MAX_LOADED_MODELS=2 (or use smaller models) and rerun.")
    elif not notes:
        notes.append(f"models resident: {models}")
    return notes


def default_llm(model: str = MODEL) -> Callable:
    def llm(messages: list[dict], schema: Optional[dict] = None) -> str:
        import ollama
        kw = dict(model=model, messages=messages, options={"temperature": 0, "num_ctx": NUM_CTX},
                  keep_alive=KEEP_ALIVE)
        if schema:
            kw["format"] = schema
        try:
            resp = ollama.chat(think=False, **kw)
        except TypeError:                                   # older client without `think`
            resp = ollama.chat(**kw)
        return resp["message"]["content"]
    return llm


class Supervisor:
    def __init__(self, store, retrieval, auditor, llm: Optional[Callable] = None, intake: Callable = None,
                 tools: Optional[dict] = None, max_steps: int = 4,
                 max_queries: int = DEFAULT_MAX_QUERIES):
        """
        store:      UserStore
        retrieval:  RetrievalAgent (anything with .run(plan, queries=None, exclude=()) -> list[Chunk])
        auditor:    Auditor        (anything with .audit(question, chunks, kept_context) -> RetrievalAudit)
        llm:        callable(messages, schema=None) -> str; defaults to local Ollama (SUPERVISOR_MODEL, qwen3.5:4b)
        intake:     callable(messages, today, prior) -> Result; defaults to decomposer.intake_and_plan
        tools:      {"compute_tax": fn(plan) -> str, ...}; offered to the model only for COMPUTATION questions (Decompose has no tools field)
        max_steps:  action budget before the supervisor must answer with what it has
        max_queries: most queries run per search (first search: the planned ones, in order; follow-ups: new ones)
        """
        self.store, self.retrieval, self.auditor = store, retrieval, auditor
        self.llm = llm or default_llm()
        self.tools = tools or {}
        self.max_steps = max_steps
        self.max_queries = max(1, max_queries)
        if intake is None:
            def intake(messages, today, prior):          # imported on first use: pulls in ollama
                from decomposer import intake_and_plan
                return intake_and_plan(messages, today=today, prior=prior)
        self.intake = intake

    # ------------------------------------------------------------ full turn
    def handle_turn(self, user_id: str, messages: list[dict], today: Optional[date] = None) -> SupervisorResult:
        t_start = time.perf_counter()
        profile, created = self.store.ensure_profile(user_id)
        stored = self.store.profile_entities(user_id)
        pending = self.store.pop_pending(user_id)         # entities from a clarification we asked last turn
        prior = _merge(stored, pending)
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")

        t_intake = time.perf_counter()
        result = self.intake(messages, today or date.today(), prior)
        intake_s = time.perf_counter() - t_intake
        out = self.supervise(user_id, result, question, reused=_set_fields(stored))
        out.timings = [("intake", intake_s), *out.timings, ("total", time.perf_counter() - t_start)]
        out.trace.insert(0, f"profile {'created' if created else 'loaded'} for {user_id}; "
                            f"saved facts: {_set_fields(stored) or 'none'}")
        out.trace.append(timing_line(out.timings))
        return out

    # ------------------------------------------------------------ planner output in
    def supervise(self, user_id: str, result, question: str, reused: Optional[dict] = None) -> SupervisorResult:
        if isinstance(result, Refuse):
            self.store.pop_pending(user_id)
            return SupervisorResult("refused", result.reason, trace=["refused by planner"])

        if isinstance(result, Clarify):
            self.store.save_pending(user_id, result.entities)   # survive the round trip even if the process restarts
            return SupervisorResult("clarify", result.question, trace=[f"clarify: missing {result.missing}"])

        plan: Decompose = result
        trace: list[str] = []
        changes = self.store.remember(user_id, plan.entities, question)
        trace.append(f"profile updates: {changes or 'none'}")

        # facts taken from storage rather than from this message are disclosed, like the planner's own assumptions
        assumptions = list(plan.assumptions)
        used = {f: v for f, v in (reused or {}).items() if getattr(plan.entities, f, None) == v
                and f not in {c[0] for c in changes}}
        if used:
            assumptions.append("Used your saved company details: "
                               + ", ".join(f"{_label(f)} {v}" for f, v in used.items())
                               + ". Tell me if any of these changed.")

        run = _Run()
        audit_q = _audit_question(plan, question)
        stop, ask = self._agent_loop(plan, question, audit_q, run, trace)

        def out(status, message, **kw):
            return SupervisorResult(status, message, plan=plan, audits=run.audits, stop_reason=stop,
                                    profile_changes=changes, trace=trace, timings=run.timings, **kw)

        if ask:
            self.store.save_pending(user_id, plan.entities)
            return out("clarify", ask)
        if not run.vetted and not run.tool_out:
            return out("no_answer", NO_CONTEXT)

        # --- the supervisor writes the final answer
        docs = self._documents(run)
        partial = stop in ("max_steps", "no_new_queries") or bool(run.audits and not run.audits[-1].all_relevant)
        try:
            draft = _timed(run.timings, "answer", self.llm,
                           self._answer_messages(question, plan, docs, assumptions, partial)).strip()
        except Exception as e:
            trace.append(f"answer model failed: {e}")
            return out("no_answer", "I couldn't generate an answer right now. Please try again.")
        trace.append(f"answered from {len(docs)} document(s)")
        if draft.startswith("INSUFFICIENT_CONTEXT") or not draft:
            return out("no_answer", NO_CONTEXT, draft=draft)
        sources = self._sources(docs, draft)
        return out("answered", draft + _footer(assumptions, sources), draft=draft, sources=sources)

    # ------------------------------------------------------------ the agent loop
    def _agent_loop(self, plan: Decompose, question: str, audit_q: str, run: _Run, trace: list[str]):
        """Returns (stop_reason, ask_user_question or None). Fills `run`."""
        available = list(self.tools) if plan.intent == Intent.COMPUTATION else []   # calculators only for computations

        for step in range(1, self.max_steps + 1):
            if step == 1:
                d = {"action": "search", "queries": [], "reason": "opening search with the planned queries"}
            else:
                d = _timed(run.timings, "decide", self._decide, question, plan, run, available, trace)
            action = d["action"]
            trace.append(f"step {step}: {action} - {d.get('reason', '')[:120]}")

            if action == "search":
                if step == 1:
                    queries = plan.queries[: self.max_queries]                # the planned queries, capped
                    if len(plan.queries) > len(queries):
                        trace.append(f"  using {len(queries)} of {len(plan.queries)} planned queries (max_queries)")
                else:
                    wanted = d.get("queries") or run.suggestions
                    flt = plan.queries[0].filter if plan.queries else SearchFilter(domain=plan.domain, year=plan.tax_year)
                    queries = [PlannedQuery(text=q, filter=flt) for q in _new_queries(wanted, run.tried)[: self.max_queries]]
                    if not queries:
                        run.log.append("search skipped: no new queries")
                        return "no_new_queries", None
                self._search(plan, audit_q, queries, run, trace)
            elif action == "run_tool":
                name = d.get("tool", "")
                if name not in available or name in run.tool_out:
                    run.log.append(f"run_tool {name!r}: not available or already run")
                    continue
                try:
                    run.tool_out[name] = str(_timed(run.timings, "tool", self.tools[name], plan))
                    run.log.append(f"run_tool {name}: done")
                except Exception as e:
                    run.log.append(f"run_tool {name}: failed ({e})")
            elif action == "ask_user":
                q = (d.get("question") or "").strip()
                if q:
                    return "agent_asked_user", q
                run.log.append("ask_user without a question ignored")
            elif action == "answer":
                if not run.vetted and not run.tool_out:
                    trace.append("answer requested with nothing vetted -> give_up")
                    return "agent_gave_up", None
                return "agent_answered", None
            else:                                                             # give_up
                return "agent_gave_up", None
        return "max_steps", None

    def _search(self, plan: Decompose, audit_q: str, queries, run: _Run, trace: list[str]) -> None:
        batch = _timed(run.timings, "retrieval", self.retrieval.run, plan, queries=queries, exclude=run.judged)
        trace += [f"  {n}" for n in getattr(self.retrieval, "last_notes", [])]
        used = [q.text for q in queries]
        run.tried += used
        batch = [c for c in batch if c.id not in run.judged]
        if not batch:
            run.log.append(f"search {used}: no new chunks")
            run.suggestions = []
            return
        kept = "\n".join(f"- {c.metadata.get('headings') or c.metadata.get('source', '')}"
                         for c in run.vetted.values())
        audit = _timed(run.timings, "audit", self.auditor.audit, audit_q, batch, kept)
        run.audits.append(audit)
        for c, ok in zip(batch, audit.relevant):
            run.judged.add(c.id)
            if ok:
                run.vetted[c.id] = c
        run.suggestions = list(audit.suggested_queries)
        n_ok = sum(audit.relevant)
        note = " (auditor unavailable: kept unjudged)" if audit.unavailable else ""
        run.log.append(f"search {used}: {n_ok}/{len(batch)} chunks related{note}; "
                       f"auditor suggests: {audit.suggested_queries or 'nothing'}")
        trace.append(f"  {run.log[-1]}")

    def _decide(self, question: str, plan: Decompose, run: _Run, available: list[str], trace: list[str]) -> dict:
        topics = [str(c.metadata.get("headings") or c.metadata.get("source", ""))[:90] for c in run.vetted.values()]
        state = (f"QUESTION: {_audit_question(plan, question)}\n"
                 f"AVAILABLE TOOLS: {[t for t in available if t not in run.tool_out] or 'none'}\n"
                 f"VETTED DOCUMENTS ({len(run.vetted)}): {topics or 'none yet'}\n"
                 f"ACTIONS SO FAR:\n" + "\n".join(f"- {l}" for l in run.log))
        try:
            raw = self.llm([{"role": "system", "content": DECIDE_SYSTEM}, {"role": "user", "content": state}],
                           DECIDE_SCHEMA)
            d = json.loads(raw)
            if d.get("action") not in DECIDE_SCHEMA["properties"]["action"]["enum"]:
                raise ValueError(f"bad action {d.get('action')!r}")
            return d
        except Exception as e:
            fallback = "answer" if run.vetted or run.tool_out else "give_up"
            trace.append(f"decision unparseable ({e}); falling back to {fallback}")
            return {"action": fallback, "reason": "fallback after unusable model output"}

    # ------------------------------------------------------------ answer step
    @staticmethod
    def _documents(run: _Run) -> list[Chunk]:
        docs = list(run.vetted.values())[:MAX_DOCS]
        docs += [Chunk(id=f"tool:{n}", content=t, metadata={"source": f"tool:{n}"}) for n, t in run.tool_out.items()]
        return docs

    @staticmethod
    def _answer_messages(question: str, plan: Decompose, docs: list[Chunk], assumptions: list[str], partial: bool):
        def head(i, d):
            where = d.metadata.get("headings")
            return f"[Doc {i}] (source: {d.metadata.get('source', '')}" + (f" | {where}" if where else "") + ")"
        body = "\n\n".join(f"{head(i, d)}\n{d.content[:DOC_CHARS]}" for i, d in enumerate(docs, 1))
        user = f"QUESTION: {question}\n\nUSER FACTS:\n{_user_facts(plan)}\n"
        if assumptions:
            user += "\nASSUMPTIONS MADE:\n" + "\n".join(f"- {a}" for a in assumptions) + "\n"
        if partial:
            user += ("\nCOVERAGE NOTE: retrieval ended before every searched chunk was confirmed related; "
                     "the documents may only cover part of the question.\n")
        user += f"\nDOCUMENTS:\n{body}"
        return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def _sources(docs: list, draft: str) -> list[str]:
        """Sources the answer cites ([Doc n]); if it cites none, every vetted source."""
        cited = {int(n) for n in re.findall(r"\[Doc (\d+)\]", draft)}
        pick = [d for i, d in enumerate(docs, 1) if i in cited] or docs
        seen, out = set(), []
        for d in pick:
            s = d.metadata.get("source", "")
            if s and s not in seen and not s.startswith("tool:"):
                seen.add(s)
                out.append(s)
        return out


# ---------------------------------------------------------------- helpers
def _new_queries(wanted: list, tried: list) -> list[str]:
    seen = {t.lower() for t in tried}
    out = []
    for q in wanted:
        q = re.sub(r"\s+", " ", str(q)).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:3]


def _entity_facts(e) -> dict:
    """The non-empty facts of any domain's entities (topics and the 'active' default are shown separately/not at all)."""
    return {f: v for f, v in e.model_dump().items()
            if v not in (None, "", []) and f not in ("topics", "company_status")}


def _user_facts(plan: Decompose) -> str:
    facts = {"domain": plan.domain.value, "year": plan.tax_year,
             **{_label(f): v for f, v in _entity_facts(plan.entities).items()}}
    lines = [f"- {k}: {v}" for k, v in facts.items() if v not in (None, "")]
    return "\n".join(lines) or "- none"


def _audit_question(plan: Decompose, question: str) -> str:
    """The prompt relevance is judged against: the question plus what we know about the user."""
    facts = [f"domain {plan.domain.value}", f"year {plan.tax_year}"] + [
        f"{_label(f)}: {v}" for f, v in _entity_facts(plan.entities).items()]
    topics = getattr(plan.entities, "topics", [])
    if topics:
        facts.append("topics: " + ", ".join(topics))
    return f"{question}\n(Known: {'; '.join(facts)})"


def _label(field_name: str) -> str:
    return {"fye": "financial year end", "fy": "financial year", "ya": "year of assessment",
            "is_new_company": "newly incorporated", "incorporation_year": "incorporation year",
            "tax_resident": "tax resident", "chargeable_income": "chargeable income (SGD)",
            "annual_revenue": "annual revenue (SGD)"}.get(field_name, field_name.replace("_", " "))


def _set_fields(e) -> dict:
    return {f: getattr(e, f) for f in PERSISTENT_FIELDS if getattr(e, f, None) is not None}


def _merge(stored: DomainEntities, pending: Optional[DomainEntities]) -> DomainEntities:
    """The planner's `prior`: saved company facts (CIT), overridden by what a clarification turn already collected."""
    if pending is None:
        return stored
    if type(pending) is not type(stored):               # e.g. a property-tax clarification: saved CIT facts don't apply
        return pending
    merged = stored.model_copy(deep=True)               # facts from the clarification turn win over stored ones
    for f, v in pending.model_dump().items():
        if f == "topics":
            merged.topics = list(dict.fromkeys([*merged.topics, *v]))
        elif v is not None:
            setattr(merged, f, v)
    return merged


def _footer(assumptions: list[str], sources: list[str]) -> str:
    out = ""
    if assumptions:
        out += "\n\nAssumptions:\n" + "\n".join(f"- {a}" for a in assumptions)
    if sources:
        out += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return out