"""Supervisor agent: a model (local qwen3.5:4b via Ollama) decides each next action and writes the final answer.

    messages -> intake gate (decomposer.intake_and_plan: extraction + deterministic gates) -> Decompose | Clarify | Refuse
                                      |
        user store (ChromaDB) <-------+  profile in as `prior`; new facts out; pending state across clarifications
                                      v
      +--------------------- supervisor model picks one action per step ----------------------+
      |  step 1: route    -> "search" the question as written, or "decompose" it first         |
      |  search(queries)  -> RetrievalAgent (wraps retriever.Retriever: hybrid + rerank)       |
      |                      -> Auditor (wraps auditor.AuditorAgent) judges each chunk          |
      |                      -> approved chunks are kept; auditor's suggested query is reported |
      |  decompose        -> query planner (decomposer.plan_queries) writes one query per fact |
      |                      (after step 1: only for the fact the auditor says is missing),     |
      |                      then those queries are searched as above, in the same step        |
      |  run_tool(name)   -> e.g. compute_tax, output becomes a citable document               |
      |  ask_user(q)      -> status "clarify" (entities saved so the next turn resumes)        |
      |  answer           -> supervisor writes the final answer from vetted chunks             |
      |  give_up          -> status "no_answer"                                                |
      +-----------------------------------------------------------------------------------------+

The intake gate only classifies and extracts (domain, intent, entities) so out-of-scope requests are refused and
missing mandatory facts are asked for before any retrieval or supervisor call. It writes no sub-queries: whether a
question is complex enough to break down is the supervisor's decision (the route step). With
Supervisor(intake_plans=True) the intake writes sub-queries in the same call as before (Decompose.planned=True), the
route step is skipped and step 1 searches those queries: the original pipeline, kept for comparison runs.

Integration layer (this file): the real components have different shapes from what the loop needs, so two small
adapters live here instead of changing them:
  * RetrievalAgent  - .run(plan, queries=None, exclude=()) -> list[Chunk], built on Retriever.search()
  * Auditor         - .audit(question, chunks, kept_context) -> RetrievalAudit, built on AuditorAgent.evaluate()
                      (via agent_judge). Any failure fails open: chunks are kept unjudged and the trace says so.

Guard rails in code (a 4B model is not trusted with the loop itself):
  * step 1 is always a search: of the planned queries, or (route step) of the question or its decomposition
  * a route or planner failure falls back to searching the question as written
  * a budget of `max_steps` actions, after which the supervisor must answer with what it has
  * `answer` with no vetted chunks and no tool output becomes give_up (never an answer from nothing)
  * invalid actions (unknown tool, repeated queries, empty ask_user) are logged and cost a step
  * unparseable model output falls back to: answer if anything is vetted, else give_up
  * follow-up queries that repeat or nearly repeat a searched one (keyword overlap >= QUERY_SIMILARITY) are
    dropped before retrieval; when none are left, the loop stops and answers from what is vetted
  * chunks the reranker scores below `audit_min_score` are rejected without an auditor call
  * when the auditor reports the approved chunks are sufficient, the loop answers without another decide call
  * when every query the auditor suggests repeats one already searched, the loop answers without a decide call

All model calls go through one injectable `llm(messages, schema=None) -> str`, the planner is an injectable
`planner(question, known, missing, tried, max_queries) -> list[str]`, and the auditor takes an injectable `judge`,
so tests need no Ollama. Two entry points:
  handle_turn(user_id, messages)  full harness: loads the profile, runs intake, supervises the result
  supervise(user_id, result, question)  takes an already-computed intake Result
"""
import functools
import inspect
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

# Take the schema classes from `decomposer` itself, not straight from decomposer_general_schemas: isinstance() checks
# and pydantic validation compare classes by identity, so they must be the very classes resolve() builds its results
# from, however decomposer.py happens to import its schemas (a second import path gives a second, "different" class).
from decomposer import (
    CITEntities, Clarify, Decompose, DomainEntities, Intent, PlannedQuery, Refuse, SearchFilter,
)
from src.retrieval.user_store import PERSISTENT_FIELDS

MODEL = os.environ.get("SUPERVISOR_MODEL", os.environ.get("INTAKE_MODEL", "qwen3.5:4b"))
# AUDITOR_MODEL = os.environ.get("AUDITOR_MODEL", "phi3.5:3.8b")   # the auditor runs its own local model by default (set it equal to SUPERVISOR_MODEL to keep one model loaded)
AUDITOR_MODEL = os.environ.get("AUDITOR_MODEL", "qwen3.5:4b")   # the auditor runs its own local model by default (set it equal to SUPERVISOR_MODEL to keep one model loaded)
NUM_CTX = 8192
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE_CLIENT", "30m")   # how long Ollama keeps our models in memory after a call

# Speed knobs: every chunk costs a reranker pass, auditor prompt tokens and answer prompt tokens, and every planned
# query costs a full retrieval (embedding + BM25 + rerank). These defaults trade a little recall for a lot of latency;
# harness.py exposes them as flags (--k, --max-chunks, --max-queries, --candidates, --audit-min-score).
DEFAULT_K = 3                       # chunks kept per query after reranking
DEFAULT_MAX_CHUNKS = 6              # chunks handed to the auditor per search
DEFAULT_MAX_QUERIES = 2             # planned / follow-up queries per search (the planner may write up to 3)
DEFAULT_CANDIDATES = 6              # chunks the reranker scores per query (config.py's CANDIDATES is 10); reranking
                                    # is the slowest part of retrieval, and k=3 rarely comes from below rank 6
AUDIT_MIN_SCORE = 0.5               # chunks the reranker scores below this are rejected without asking the auditor
                                    # (every audited chunk costs LLM output tokens); 0 sends everything to the auditor
QUERY_SIMILARITY = 0.7              # a follow-up query sharing this much of its keywords (Jaccard) with one already
                                    # searched, or adding no keyword not searched before, is a near-duplicate: it
                                    # retrieves the same chunks, so it is not run
DOC_CHARS = 1200                   # per-chunk text shown to the model when answering
AUDIT_CHARS = 800                   # per-chunk text shown to the auditor
MAX_DOCS = 6                        # vetted chunks put in the answer prompt
MAX_DRY_ROUNDS = 2                  # stop after this many searches in a row that approve nothing new (and nothing is vetted)

NO_CONTEXT = "The IRAS sources I have don't cover this. Please check iras.gov.sg or rephrase your question."

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "decompose"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

ROUTE_SYSTEM = """You route questions for a Singapore corporate income tax assistant. Decide how to search the IRAS
documents for the QUESTION before anything has been searched.
- search: the question asks for ONE fact, rate, rule, deadline or procedure. Searching the question as written will
  find it. Most questions are like this.
- decompose: answering needs SEVERAL separate facts that sit in different places, e.g. several eligibility conditions,
  a rate plus a rebate or exemption, a comparison of two schemes, a computation, or one fact that depends on another.
  A planner will then write one search query per fact.
Reply with JSON: {"action": "search" or "decompose", "reason": "..."}"""

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "decompose", "run_tool", "ask_user", "answer", "give_up"]},
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
- decompose: a planner writes one targeted query for each fact the auditor reports as missing, and they are searched.
  Use it when part of the question is still uncovered and you cannot name a good query yourself.
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
    missing: Optional[str] = None              # auditor's name for the fact still missing, when not sufficient
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

    def warm(self, query: str = "corporate income tax rate") -> str:
        """Run one throwaway search so the reranker is loaded (it loads lazily on first use) and the embedder has
        run once, before the first real question pays for both. Returns a note for the console. Never raises."""
        t0 = time.perf_counter()
        try:
            extra = {} if self.candidates is None else {"candidates": self.candidates}
            self.retriever.search(query, k=1, **extra)
        except Exception as e:
            return f"could not warm the retriever: {e}"
        return f"retriever warm ({time.perf_counter() - t0:.1f}s)"

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
    """Build a judge(question, chunks, kept_context, tried=(), year=None) -> dict on top of AuditorAgent.evaluate().

    `agent` is anything with .evaluate(original_query, sub_prompt, chunks, tried_queries=..., year=...) returning an
    object with is_sufficient, missing, recommended_search_query and chunk_evaluations[chunk, status, reason], where
    `chunk` is the 1-based position in `chunks` (an older agent that returns `chunk_id` instead is still understood).
    When omitted, a real AuditorAgent is created on first use against the local Ollama server (so importing this
    module needs no running Ollama).
    """
    holder = {"agent": agent}

    def judge(question: str, chunks: list, kept_context: str, tried=(), year=None) -> dict:
        if holder["agent"] is None:
            from src.auditor.auditor import AuditorAgent
            # qwen3.x are thinking models and would be slow here; other models (phi, gemma) don't take the flag
            holder["agent"] = AuditorAgent(model=model, keep_alive=KEEP_ALIVE,
                                           think=False if model.startswith("qwen3") else None)
        sub_prompt = question
        if kept_context.strip():
            sub_prompt += f"\n(Topics already covered by approved chunks:\n{kept_context})"
        payload = [{"id": c.id, "headings": c.metadata.get("headings", ""), "text": c.content[:AUDIT_CHARS]}
                   for c in chunks]
        resp = holder["agent"].evaluate(original_query=question, sub_prompt=sub_prompt, chunks=payload,
                                        tried_queries=list(tried), year=year)
        position = {c.id: i for i, c in enumerate(chunks, 1)}
        by_n = {}
        for e in resp.chunk_evaluations:
            n = getattr(e, "chunk", None)
            if n is None:                                   # older auditor that copies chunk ids
                n = position.get(str(getattr(e, "chunk_id", "")).strip())
            if isinstance(n, int) and 1 <= n <= len(chunks) and n not in by_n:
                by_n[n] = e                                 # first verdict wins if the model repeats a number
        rows = []
        for i in range(1, len(chunks) + 1):
            e = by_n.get(i)
            if e is None:       # the small model skipped this chunk: keep it rather than silently lose evidence
                rows.append({"index": i, "relevant": True, "reason": "not evaluated by auditor"})
            else:
                reason = str(e.reason or "").strip()
                if reason.strip('"').lower() in ("null", "none"):     # the model wrote null as a word
                    reason = ""
                rows.append({"index": i, "relevant": e.status == "approved", "reason": reason})
        def text(v):                                        # "null"/"none" written as a word counts as absent
            v = str(v or "").strip()
            return "" if v.strip('"').lower() in ("null", "none") else v
        suggested = [text(resp.recommended_search_query)] if text(resp.recommended_search_query) else []
        return {"chunks": rows, "suggested_queries": suggested, "sufficient": resp.is_sufficient,
                "missing": text(getattr(resp, "missing", None)) or None}

    return judge


def _takes(fn: Callable, name: str) -> bool:
    """Whether fn accepts keyword argument `name` (test judges and older custom judges take only 3 arguments)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class Auditor:
    """Judges which retrieved chunks are related to the question. Fails open (keeps everything) if the judge breaks."""

    def __init__(self, judge: Optional[Callable] = None, model: str = AUDITOR_MODEL):
        self.judge = judge or agent_judge(model=model)
        self._extra = [k for k in ("tried", "year") if _takes(self.judge, k)]

    def audit(self, question: str, chunks: list, kept_context: str = "", tried=(), year=None) -> RetrievalAudit:
        """tried: queries already searched (the auditor is told not to suggest them again); year: the YA asked."""
        n = len(chunks)
        extra = {k: v for k, v in (("tried", list(tried)), ("year", year)) if k in self._extra}
        try:
            verdict = self.judge(question, chunks, kept_context, **extra)
            rows = {int(r["index"]): r for r in verdict.get("chunks", [])}
            relevant = [bool(rows[i]["relevant"]) if i in rows else True for i in range(1, n + 1)]
            reasons = [str(rows[i].get("reason") or "") if i in rows else "" for i in range(1, n + 1)]
            suggested = [re.sub(r"\s+", " ", str(q)).strip() for q in verdict.get("suggested_queries", [])]
            missing = re.sub(r"\s+", " ", str(verdict.get("missing") or "")).strip() or None
            return RetrievalAudit(relevant=relevant, reasons=reasons, suggested_queries=[q for q in suggested if q],
                                  sufficient=verdict.get("sufficient"), missing=missing)
        except Exception as e:
            return RetrievalAudit(relevant=[True] * n, reasons=[f"auditor unavailable: {e}"] * n, unavailable=True)


@dataclass
class SupervisorResult:
    status: str                         # answered | clarify | refused | no_answer
    message: str                        # what to show the user
    plan: Optional[Decompose] = None
    draft: str = ""                     # raw answer from the model
    audits: list[RetrievalAudit] = field(default_factory=list)   # one per search action
    stop_reason: str = ""               # agent_answered | auditor_sufficient | suggestions_repeat | agent_gave_up | max_steps | no_new_queries | ...
    route: str = ""                     # step 1: "planned" (intake wrote the queries) | "search" | "decompose"
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
    dry: int = 0                                        # consecutive searches that approved nothing
    tried: list = field(default_factory=list)           # queries already searched
    log: list = field(default_factory=list)             # one line per action, shown back to the model
    suggestions: list = field(default_factory=list)     # auditor's latest suggested queries
    seen_chunk_ids: set = field(default_factory=set)    # every chunk id any search returned
    last_new_chunks_count: Optional[int] = None         # chunks the latest search returned for the first time
    route: str = ""                                     # how step 1 searched (see SupervisorResult.route)


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
            # same num_ctx as the real calls: Ollama treats a different context size as a different load
            ollama.generate(model=m, prompt="", keep_alive=keep_alive, options={"num_ctx": NUM_CTX})
        except Exception as e:
            notes.append(f"could not load {m}: {e}")
    try:
        resp = ollama.ps()
        items = getattr(resp, "models", None)
        if items is None:
            items = resp.get("models", [])
        loaded = {}
        for it in items:
            name = getattr(it, "model", None) or (it.get("model") if hasattr(it, "get") else None)
            size = getattr(it, "size", None) or (it.get("size") if hasattr(it, "get") else None)
            if name:
                loaded[name] = f"{name} ({size / 1e9:.1f} GB)" if size else name
    except Exception as e:
        return notes + [f"could not check which models are loaded: {e}"]
    missing = [m for m in models if m not in loaded and not any(n.startswith(m) for n in loaded)]
    if missing:
        notes.append(
            f"{missing} did not stay loaded; resident now: {list(loaded.values()) or 'nothing'}. Ollama evicted it "
            "when the next model loaded: either both don't fit in memory together, or the server runs with "
            "OLLAMA_MAX_LOADED_MODELS=1. Check the SIZE column of `ollama ps`. The simplest fix is one shared model "
            "for every part, e.g. AUDITOR_MODEL=qwen3.5:4b (same as SUPERVISOR_MODEL / INTAKE_MODEL).")
    elif not notes:
        notes.append(f"models resident: {list(loaded.values())}")
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


def default_planner(question: str, known: str = "", missing: Optional[str] = None, tried=(),
                    max_queries: int = 3) -> list[str]:
    from decomposer import plan_queries                      # imported on first use: pulls in ollama
    return plan_queries(question, known=known, missing=missing, tried=tried, max_queries=max_queries)


class Supervisor:
    def __init__(self, store, retrieval, auditor, llm: Optional[Callable] = None, intake: Callable = None,
                 tools: Optional[dict] = None, max_steps: int = 4,
                 max_queries: int = DEFAULT_MAX_QUERIES, audit_min_score: float = AUDIT_MIN_SCORE,
                 planner: Optional[Callable] = None, intake_plans: bool = False):
        """
        store:      UserStore
        retrieval:  RetrievalAgent (anything with .run(plan, queries=None, exclude=()) -> list[Chunk])
        auditor:    Auditor        (anything with .audit(question, chunks, kept_context) -> RetrievalAudit)
        llm:        callable(messages, schema=None) -> str; defaults to local Ollama (SUPERVISOR_MODEL, qwen3.5:4b).
                    Used for the route step, every decide step and the answer.
        intake:     callable(messages, today, prior) -> Result; defaults to decomposer.intake_and_plan
        tools:      {"compute_tax": fn(plan) -> str, ...}; offered to the model only for COMPUTATION questions (Decompose has no tools field)
        max_steps:  action budget before the supervisor must answer with what it has
        max_queries: most queries run per search (first search: the planned ones, in order; follow-ups: new ones)
        audit_min_score: chunks with a reranker score below this are rejected without an auditor call (0 = off)
        planner:    callable(question, known, missing, tried, max_queries) -> list[str]; defaults to
                    decomposer.plan_queries. Runs only when the supervisor chooses `decompose`.
        intake_plans: True = the default intake also writes sub-queries (the original pipeline, no route step)
        """
        self.store, self.retrieval, self.auditor = store, retrieval, auditor
        self.llm = llm or default_llm()
        self.tools = tools or {}
        self.max_steps = max_steps
        self.max_queries = max(1, max_queries)
        self.audit_min_score = audit_min_score or 0.0
        self.planner = planner or default_planner
        self.intake_plans = intake_plans
        if intake is None:
            def intake(messages, today, prior):          # imported on first use: pulls in ollama
                from decomposer import intake_and_plan
                return intake_and_plan(messages, today=today, prior=prior, plan_queries=self.intake_plans)
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

    # ------------------------------------------------------------ intake output in
    def supervise(self, user_id: str, result, question: str, reused: Optional[dict] = None) -> SupervisorResult:
        if isinstance(result, Refuse):
            self.store.pop_pending(user_id)
            return SupervisorResult("refused", result.reason, trace=["refused by intake gate"])

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
                                    route=run.route, profile_changes=changes, trace=trace, timings=run.timings, **kw)

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
                if plan.planned:                            # the intake already decomposed it: search that plan
                    run.route = "planned"
                    d = {"action": "search", "queries": [], "reason": "opening search with the planned queries"}
                else:                                       # the supervisor decides: as written, or broken down
                    d = _timed(run.timings, "route", self._route, plan, question, trace)
                    run.route = d["action"]
            else:
                # The last search's audit approved something but the auditor's only ideas for what is missing are
                # queries already searched (or near-repeats): another search would bring back the same chunks, so
                # answer now instead of paying for a decide call whose search would be skipped anyway.
                repeats = _repeated_suggestions(run, available)
                if repeats:
                    trace.append(f"step {step}: answer - auditor's suggestions {repeats} repeat searched queries; "
                                 "skipping the decide call")
                    return "suggestions_repeat", None
                d = _timed(run.timings, "decide", self._decide, question, plan, run, available, trace)

            action = d["action"]
            trace.append(f"step {step}: {action} - {d.get('reason', '')[:120]}")

            if action in ("search", "decompose"):
                # 1. Prepare queries first
                queries = self._next_queries(step, action, d, plan, question, run, trace)
                if queries is None:
                    run.log.append("search skipped: no new queries")
                    trace.append("  guard: no new queries left to search; stopping search loop.")
                    return "no_new_queries", None          # supervise() answers if anything is vetted
                approved_before = len(run.vetted)

                # 2. Execute search ONCE
                audit = self._search(plan, audit_q, queries, run, trace)

                # 3. Early termination on zero-delta
                # Only trigger on step > 1 (step 1 is the initial cold search)
                # eliminates redundant multi-hop loops. speeding up time for retrieval
                if step > 1 and run.last_new_chunks_count == 0:
                    trace.append("  guard: zero new chunks retrieved; stopping search loop.")
                    if run.vetted or run.tool_out:
                        return "agent_answered", None
                    return "agent_gave_up", None

                # 4. The auditor says the approved chunks answer every part of the question: answer now instead of
                # paying for another decide call (unless a calculator still has to run for a computation).
                pending_tools = [t for t in available if t not in run.tool_out]
                if audit is not None and audit.sufficient is True and run.vetted and not pending_tools:
                    trace.append("  guard: auditor reports the approved chunks are sufficient; answering.")
                    return "auditor_sufficient", None

                run.dry = 0 if len(run.vetted) > approved_before else run.dry + 1
                if run.dry >= MAX_DRY_ROUNDS and not run.vetted and not run.tool_out:
                    trace.append(f"  {run.dry} searches in a row found nothing usable: stopping early")
                    return "no_progress", None
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

            else:  # give_up
                return "agent_gave_up", None

        return "max_steps", None

    def _next_queries(self, step: int, action: str, d: dict, plan: Decompose, question: str, run: _Run,
                      trace: list[str]):
        """The PlannedQuery list this search/decompose step runs, or None when nothing new is left to search."""
        base = plan.queries[0] if plan.queries else None
        flt = base.filter if base else SearchFilter(domain=plan.domain.value, year=plan.tax_year)
        make = type(base) if base else PlannedQuery  # build with the plan's own classes
        skipped: list[str] = []

        if action == "decompose":
            # Step 1: break the whole question down. Later: plan only for what the last audit says is missing.
            missing = run.audits[-1].missing if step > 1 and run.audits else None
            drafted = _timed(run.timings, "plan", self._plan, plan, question, missing, run.tried, trace)
            texts = _new_queries(drafted, run.tried, skipped)
            if skipped:
                trace.append(f"  guard: skipped near-duplicate planned queries {skipped}")
            if texts:
                trace.append(f"  planner queries{f' for missing {missing!r}' if missing else ''}: {texts}")
                return [make(text=q, filter=flt) for q in texts[: self.max_queries]]
            trace.append("  planner gave no new queries; falling back to "
                         + ("the question as written" if step == 1 else "the model's and auditor's queries"))
            skipped = []

        if step == 1:
            # capping queries limits total database round-trips
            queries = plan.queries[: self.max_queries]
            if len(plan.queries) > len(queries):
                trace.append(f"  using {len(queries)} of {len(plan.queries)} planned queries (max_queries)")
            return queries

        # Repeats and near-repeats of searched queries retrieve the same chunks, so they are dropped
        # before any retrieval runs. If none of the model's queries are new, the auditor's get a chance.
        texts = (_new_queries(d.get("queries") or [], run.tried, skipped)
                 or _new_queries(run.suggestions, run.tried, skipped))
        if skipped:
            trace.append(f"  guard: skipped near-duplicate queries {skipped}")
        queries = [make(text=q, filter=flt) for q in texts[: self.max_queries]]
        return queries or None

    def _route(self, plan: Decompose, question: str, trace: list[str]) -> dict:
        """Step 1 when the intake wrote no sub-queries: search the question as written, or decompose it first.
        Any failure falls back to searching as written (the cheaper path, and what a simple question needs)."""
        state = (f"QUESTION: {_ask(plan, question)}\n"
                 f"INTENT: {plan.intent.value}\n"
                 f"KNOWN: {_known(plan)}")
        try:
            raw = self.llm([{"role": "system", "content": ROUTE_SYSTEM}, {"role": "user", "content": state}],
                           ROUTE_SCHEMA)
            d = json.loads(raw)
            if d.get("action") not in ROUTE_SCHEMA["properties"]["action"]["enum"]:
                raise ValueError(f"bad route {d.get('action')!r}")
            return {"action": d["action"], "queries": [], "reason": str(d.get("reason") or "")}
        except Exception as e:
            trace.append(f"route unparseable ({e}); searching the question as written")
            return {"action": "search", "queries": [], "reason": "fallback after unusable route output"}

    def _plan(self, plan: Decompose, question: str, missing: Optional[str], tried: list, trace: list[str]) -> list:
        """The planner's queries, or [] (and a trace line) if it fails."""
        try:
            return [str(q) for q in (self.planner(_ask(plan, question), _known(plan), missing, list(tried),
                                                  self.max_queries) or [])]
        except Exception as e:
            trace.append(f"  planner failed: {e}")
            return []

    def _search(self, plan: Decompose, audit_q: str, queries, run: _Run, trace: list[str]):
        """Retrieve, drop low-scoring chunks, audit the rest. Returns this search's RetrievalAudit, or None if no
        auditor call was made (nothing new retrieved, or everything new fell below the reranker cutoff)."""
        batch = _timed(run.timings, "retrieval", self.retrieval.run, plan, queries=queries, exclude=run.judged)
        # chunks never returned before in this run: zero of them on a follow-up search ends the loop
        run.last_new_chunks_count = sum(1 for c in batch if c.id not in run.seen_chunk_ids)
        run.seen_chunk_ids.update(c.id for c in batch)

        trace += [f"  {n}" for n in getattr(self.retrieval, "last_notes", [])]
        used = [q.text for q in queries]
        run.tried += used
        batch = [c for c in batch if c.id not in run.judged]
        if not batch:
            run.log.append(f"search {used}: no new chunks")
            run.suggestions = []
            return None

        # Reranker pre-filter: a chunk the cross-encoder scores this low is almost never approved, and every chunk
        # handed to the auditor costs generated tokens. Safety net: while nothing is approved yet, the best of
        # them is still audited, so a uniformly low-scoring topic can't end the search with no evidence at all.
        cutoff = self.audit_min_score
        below = [c for c in batch if cutoff and _score(c) is not None and _score(c) < cutoff]
        if below and len(below) == len(batch) and not run.vetted:
            below.remove(max(below, key=_score))
        below_ids = {c.id for c in below}
        batch = [c for c in batch if c.id not in below_ids]
        for c in below:
            run.judged.add(c.id)
        skipped_lines = [f"    skipped [{_score(c):.2f}] {_where(c)}: reranker score below {cutoff:.2f}, not audited"
                         for c in below]
        if not batch:
            run.log.append(f"search {used}: {len(below)} chunk(s) below reranker cutoff {cutoff:.2f}, none audited")
            trace.append(f"  {run.log[-1]}")
            trace += skipped_lines
            run.suggestions = []
            return None

        kept = "\n".join(f"- {c.metadata.get('headings') or c.metadata.get('source', '')}"
                         for c in run.vetted.values())
        audit = _timed(run.timings, "audit", self.auditor.audit, audit_q, batch, kept,
                       tried=run.tried, year=plan.tax_year)
        run.audits.append(audit)
        for c, ok in zip(batch, audit.relevant):
            run.judged.add(c.id)
            if ok:
                run.vetted[c.id] = c
        run.suggestions = list(audit.suggested_queries)
        n_ok = sum(audit.relevant)
        why = (audit.reasons[0] if audit.reasons else "").replace("auditor unavailable: ", "")[:300]
        note = f" (auditor unavailable, kept unjudged: {why})" if audit.unavailable else ""
        cut = f", {len(below)} below reranker cutoff" if below else ""
        run.log.append(f"search {used}: {n_ok}/{len(batch)} chunks related{cut}{note}; "
                       + (f"missing: {audit.missing}; " if audit.missing else "")
                       + f"auditor suggests: {audit.suggested_queries or 'nothing'}")
        trace.append(f"  {run.log[-1]}")
        if not audit.unavailable:                                   # why each chunk was kept or dropped
            reasons = list(audit.reasons) + [""] * len(batch)
            for c, ok, why in zip(batch, audit.relevant, reasons):
                score = _score(c)
                verdict = "approved" if ok else "REJECTED"
                shown = f" [{score:.2f}]" if score is not None else ""
                trace.append(f"    {verdict}{shown} {_where(c)}" + (f": {why[:200]}" if why else ""))
        trace += skipped_lines
        return audit

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
_STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "is", "are", "what", "how", "my", "our",
              "up", "with", "by", "under", "does", "do", "can", "singapore", "sg", "iras"}


# words a small model swaps freely when rephrasing "what are the conditions for X"; they retrieve the same chunks
_SYNONYMS = {"requirement": "condition", "criteria": "condition", "criterion": "condition",
             "eligibility": "condition", "eligible": "condition", "qualifying": "condition", "qualify": "condition",
             "rule": "condition"}


def _query_terms(q: str) -> set[str]:
    """The keywords a query searches on: lowercased words, minus stopwords and bare numbers ("50%", "2026"),
    with a plural 's' trimmed ("requirements" -> "requirement") and _SYNONYMS folded together."""
    terms = set()
    for w in re.findall(r"[a-z0-9]+", q.lower()):
        if w in _STOPWORDS or w.isdigit():
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        terms.add(_SYNONYMS.get(w, w))
    return terms


def _near_duplicate(q: str, others: list[str], threshold: float = QUERY_SIMILARITY) -> Optional[str]:
    """The query in `others` that `q` repeats, else None. A repeat is: the same text; keyword overlap (Jaccard)
    >= threshold with one query ("... shareholder conditions" after "... shareholder requirements"); or no keyword
    that none of `others` searched ("... requirements 50%", "tax rate" after "corporate income tax rate").
    One swapped keyword in a short query is NOT a repeat: "corporate income tax rebate" after "... tax rate" is 0.6."""
    a = _query_terms(q)
    searched: set = set()
    for o in others:
        if q.lower() == o.lower():
            return o
        b = _query_terms(o)
        searched |= b
        if a and b and len(a & b) / len(a | b) >= threshold:
            return o
    if a and others and a <= searched:
        return next(o for o in others if a & _query_terms(o))
    return None


def _new_queries(wanted: list, tried: list, skipped: Optional[list] = None) -> list[str]:
    """`wanted`, minus blanks, repeats of `tried` and near-repeats (of `tried` or of each other). Dropped queries
    are appended to `skipped` when given, so the trace can show what the guard removed."""
    seen = [str(t) for t in tried]
    out = []
    for q in wanted:
        q = re.sub(r"\s+", " ", str(q)).strip()
        if not q:
            continue
        if _near_duplicate(q, seen) is not None:
            if skipped is not None and q not in skipped:
                skipped.append(q)
            continue
        seen.append(q)
        out.append(q)
    return out[:3]


def _repeated_suggestions(run: _Run, available: list[str]) -> list[str]:
    """The auditor's suggestions when ALL of them repeat queries already searched, else []. Only counts when the
    latest action was an audited search (suggestions are cleared when a search audits nothing), something is
    vetted to answer from, and no calculator is still waiting to run for a computation question."""
    if not run.suggestions or not run.vetted or [t for t in available if t not in run.tool_out]:
        return []
    if _new_queries(run.suggestions, run.tried):
        return []
    return list(run.suggestions)


def _score(c: Chunk) -> Optional[float]:
    s = c.metadata.get("score")
    return float(s) if isinstance(s, (int, float)) else None


def _where(c: Chunk) -> str:
    return str(c.metadata.get("headings") or c.metadata.get("source", ""))[:70]


def _entity_facts(e) -> dict:
    """The non-empty facts of any domain's entities (topics and the 'active' default are shown separately/not at all)."""
    return {f: v for f, v in e.model_dump().items()
            if v not in (None, "", []) and f not in ("topics", "company_status")}


def _user_facts(plan: Decompose) -> str:
    facts = {"domain": plan.domain.value, "year": plan.tax_year,
             **{_label(f): v for f, v in _entity_facts(plan.entities).items()}}
    lines = [f"- {k}: {v}" for k, v in facts.items() if v not in (None, "")]
    return "\n".join(lines) or "- none"


def _known(plan: Decompose) -> str:
    """What we know about the user, on one line: 'domain cit; year 2026; financial year end: 2025-12-31; topics: ...'"""
    facts = [f"domain {plan.domain.value}", f"year {plan.tax_year}"] + [
        f"{_label(f)}: {v}" for f, v in _entity_facts(plan.entities).items()]
    topics = getattr(plan.entities, "topics", [])
    if topics:
        facts.append("topics: " + ", ".join(topics))
    return "; ".join(facts)


def _ask(plan: Decompose, question: str) -> str:
    """The question the route step and the planner work on. Without a planned decomposition, the intake's one query
    is every user turn of the conversation, so a clarification answer ("120,000") still carries its question."""
    if not plan.planned and plan.queries and plan.queries[0].text.strip():
        return plan.queries[0].text
    return question


def _audit_question(plan: Decompose, question: str) -> str:
    """The prompt relevance is judged against: the question plus what we know about the user."""
    return f"{question}\n(Known: {_known(plan)})"


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