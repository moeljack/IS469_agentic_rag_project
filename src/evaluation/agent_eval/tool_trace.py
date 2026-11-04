"""Record the agent's tool calls for tool-call accuracy (Eval Plan: "Tool-Call Accuracy"), without changing the agent.

The supervisor (src/supervisor/supervisor.py) logs its trace as text and does not keep the raw model output of its
tool calls, so ToolCallRecorder attaches to a running AgenticSystem (rag_eval_approach/systems.py) by wrapping
attributes of its supervisor instance, the same way systems.py and injection.py already do:

    sup.llm            every route / decide call: the raw reply and the JSON schema it was asked to follow
    sup._route         step 1, search the question as written or decompose it      -> "decision" event
    sup._decide        each later step's action and arguments                         -> "decision" event
    sup._plan          the query planner (decompose)                                  -> "plan" event
    sup._next_queries  the queries a step proposed, and which the near-duplicate guard dropped -> "queries" event
    sup._search        the executed retrieval + audit: new chunks, approved chunks, errors      -> "search" event

Events, in the order they happened (one list per question, saved as the record's `tool_calls`):
    decision  {type, kind: route|decide, step, raw, json_ok, schema_ok, schema_errors, semantic_issues, fallback,
               action, queries, tool, question, llm_error}
              raw/json_ok/schema_ok describe what the model emitted; action is what the loop then did (after a harness
              fallback, the fallback's action). fallback = the harness replaced unusable output.
    plan      {type, step, queries, failed}
    queries   {type, step, action, by: model|guard, source, proposed, kept, blocked, none_left}
              by = guard: the harness chose this step itself (the auditor's missing fact), no decide call was made
    search    {type, step, queries, new_chunks, approved, audited, auditor_unavailable, fault, error}

FaultyRetriever injects a failure into the first search of every question, to measure error recovery:
    empty   the retriever returns nothing (a tool that ran but found nothing)
    error   the retriever raises (a tool execution error)
"""
import ast
import json
import re

STEP_LINE = re.compile(r"^step (\d+):")
BLOCKED_LINE = re.compile(r"guard: skipped near-duplicate (?:planned )?queries (\[.*\])\s*$")


# ---------------------------------------------------------------- fault injection
class FaultyRetriever:
    """Wraps the evaluation retriever. While `active`, search() returns nothing (mode "empty") or raises (mode
    "error"). ToolCallRecorder switches it on for the first search of each question only."""
    MODES = ("empty", "error")

    def __init__(self, inner, mode):
        if mode not in self.MODES:
            raise ValueError(f"fault mode must be one of {self.MODES}, not {mode!r}")
        self._inner, self.mode, self.active, self.triggered = inner, mode, False, 0

    def __getattr__(self, name):      # mode, collection, _rerank, warm-up helpers... all from the real retriever
        return getattr(self._inner, name)

    def search(self, query, k=5, **kwargs):
        if self.active:
            self.triggered += 1
            if self.mode == "empty":
                return []
            raise RuntimeError("injected retrieval failure (agent_eval tool-call fault test)")
        return self._inner.search(query, k=k, **kwargs)


# ---------------------------------------------------------------- schema checks
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}


def schema_errors(value, schema, path="$"):
    """Problems with `value` against a (simple) JSON schema: type, enum, required, properties, items. [] = valid."""
    errors = []
    want = schema.get("type")
    if want in _TYPES:
        ok = isinstance(value, _TYPES[want]) and not (want in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return [f"{path}: expected {want}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if want == "object":
        errors += [f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value]
        for k, sub in (schema.get("properties") or {}).items():
            if k in value:
                errors += schema_errors(value[k], sub, f"{path}.{k}")
    if want == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def _call_kind(schema):
    """route | decide | other, from the schema the supervisor sent (ROUTE_SCHEMA / DECIDE_SCHEMA / none = answer)."""
    if not isinstance(schema, dict):
        return "other"
    enum = (((schema.get("properties") or {}).get("action") or {}).get("enum")) or []
    if "run_tool" in enum:
        return "decide"
    if set(enum) == {"search", "decompose"}:
        return "route"
    return "other"


def _last_step(trace):
    steps = [int(m.group(1)) for line in trace for m in [STEP_LINE.match(line)] if m]
    return max(steps, default=0)


def _norm(q):
    return re.sub(r"\s+", " ", str(q)).strip().lower()


# ---------------------------------------------------------------- the recorder
class ToolCallRecorder:
    """Attach to an AgenticSystem. Call reset() before each question; read .events after it."""

    def __init__(self, system, faulty=None):
        self.events, self._llm_log, self._step, self._searches = [], [], 0, 0
        self.faulty = faulty
        sup = getattr(system, "sup", None)
        self.enabled = sup is not None             # the naive system has no tool calls to record
        if self.enabled:
            self._attach(sup)

    def reset(self):
        self.events, self._llm_log, self._step, self._searches = [], [], 0, 0
        if self.faulty is not None:
            self.faulty.active = False

    # ------------------------------------------------------------ wrapping
    def _attach(self, sup):
        llm, route, decide = sup.llm, sup._route, sup._decide
        plan, next_queries, search = sup._plan, sup._next_queries, sup._search

        def rec_llm(messages, schema=None):
            entry = {"kind": _call_kind(schema), "schema": schema, "raw": None, "error": None}
            self._llm_log.append(entry)
            try:
                entry["raw"] = llm(messages, schema)
            except Exception as e:                # recorded, then handled by the supervisor as before
                entry["error"] = f"{type(e).__name__}: {e}"
                raise
            return entry["raw"]

        def rec_route(plan_, question, trace):
            n = len(self._llm_log)
            d = route(plan_, question, trace)
            self._decision("route", 1, n, d, available=[], run=None)
            return d

        def rec_decide(question, plan_, run, available, trace):
            n, step = len(self._llm_log), _last_step(trace) + 1
            d = decide(question, plan_, run, available, trace)
            self._decision("decide", step, n, d, available=list(available), run=run)
            return d

        def rec_plan(plan_, question, missing, tried, trace):
            n = len(trace)
            out = plan(plan_, question, missing, tried, trace)
            failed = any("planner failed" in line for line in trace[n:])
            self.events.append({"type": "plan", "step": self._step, "queries": [str(q) for q in out or []],
                                "failed": failed})
            return out

        def rec_next_queries(step, action, d, plan_, question, run, trace):
            self._step, n = step, len(trace)
            out = next_queries(step, action, d, plan_, question, run, trace)
            blocked = []
            for line in trace[n:]:
                m = BLOCKED_LINE.search(line)
                if m:                                # the guard prints a Python list repr
                    try:
                        blocked += [str(q) for q in ast.literal_eval(m.group(1))]
                    except (ValueError, SyntaxError):
                        blocked.append(m.group(1))
            kept = [q.text for q in out] if out is not None else []
            proposed = [str(q) for q in (d.get("queries") or [])]
            last = next((e for e in reversed(self.events) if e["type"] == "decision"), None)
            by = "model" if last is not None and last["step"] == step else "guard"
            if step == 1 and action == "search":
                source = "question"                  # step 1 searches the question as written
            elif action == "decompose":
                source = "planner"
            elif kept and {_norm(q) for q in kept} & {_norm(q) for q in proposed}:
                source = "model"
            else:
                source = "auditor" if kept else "none"
            self.events.append({"type": "queries", "step": step, "action": action, "by": by, "source": source,
                                "proposed": proposed, "kept": kept, "blocked": blocked, "none_left": out is None})
            return out

        def rec_search(plan_, audit_q, queries, run, trace):
            vetted_before = len(run.vetted)
            self._searches += 1
            fault = None
            if self.faulty is not None and self._searches == 1:
                self.faulty.active, fault = True, self.faulty.mode
            event = {"type": "search", "step": self._step, "queries": [q.text for q in queries], "new_chunks": 0,
                     "approved": 0, "audited": False, "auditor_unavailable": False, "fault": fault, "error": None}
            self.events.append(event)
            try:
                audit = search(plan_, audit_q, queries, run, trace)
            except Exception as e:
                event["error"] = f"{type(e).__name__}: {e}"
                raise
            finally:
                if self.faulty is not None:
                    self.faulty.active = False
            event.update(new_chunks=run.last_new_chunks_count or 0, approved=len(run.vetted) - vetted_before,
                         audited=audit is not None, auditor_unavailable=bool(audit is not None and audit.unavailable))
            return audit

        sup.llm, sup._route, sup._decide = rec_llm, rec_route, rec_decide
        sup._plan, sup._next_queries, sup._search = rec_plan, rec_next_queries, rec_search

    # ------------------------------------------------------------ decision events
    def _decision(self, kind, step, llm_start, d, available, run):
        calls = [c for c in self._llm_log[llm_start:] if c["kind"] == kind]
        call = calls[-1] if calls else {"raw": None, "schema": None, "error": "no model call recorded"}
        raw, schema = call.get("raw"), call.get("schema")
        parsed, json_ok = None, False
        try:
            parsed = json.loads(raw) if raw is not None else None
            json_ok = isinstance(parsed, dict)
        except (json.JSONDecodeError, TypeError):
            pass
        errors = schema_errors(parsed, schema) if json_ok and schema else (["reply is not a JSON object"] if raw is not None
                                                                          else [call.get("error") or "no reply"])
        schema_ok = json_ok and not errors
        issues = semantic_issues(parsed, kind, available, run) if schema_ok else []
        fallback = str(d.get("reason") or "").startswith("fallback after unusable")
        self.events.append({
            "type": "decision", "kind": kind, "step": step, "raw": None if raw is None else str(raw)[:2000],
            "json_ok": json_ok, "schema_ok": schema_ok, "schema_errors": errors, "semantic_issues": issues,
            "fallback": fallback, "action": d.get("action"),
            "queries": [str(q) for q in (d.get("queries") or [])], "tool": d.get("tool") or "",
            "question": d.get("question") or "", "llm_error": call.get("error")})


def semantic_issues(d, kind, available, run):
    """Arguments that are schema-valid but cannot work for the chosen action (the supervisor logs these as invalid
    actions and spends a step on them)."""
    if kind != "decide" or not isinstance(d, dict):
        return []
    action, issues = d.get("action"), []
    if action == "search" and not [q for q in d.get("queries") or [] if isinstance(q, str) and q.strip()]:
        issues.append("search without queries")
    if action == "search" and len(d.get("queries") or []) > 3:
        issues.append("more than 3 queries")
    if action == "run_tool":
        done = getattr(run, "tool_out", {}) or {}
        if d.get("tool") not in available:
            issues.append(f"tool {d.get('tool')!r} not available")
        elif d.get("tool") in done:
            issues.append(f"tool {d.get('tool')!r} already run")
    if action == "ask_user" and not str(d.get("question") or "").strip():
        issues.append("ask_user without a question")
    return issues