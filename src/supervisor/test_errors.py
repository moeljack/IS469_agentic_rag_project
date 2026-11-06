"""Scenario tests for the give_up and error-handling changes. No Ollama, ChromaDB or retriever needed: the
retriever, auditor, user store and model are stubs, and the question plan is a plain object with the attributes the
supervisor reads, so the real pydantic Decompose schema does not matter.

    python3 src/supervisor/test_errors.py
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/supervisor
import supervisor as S  # noqa: E402

@dataclass
class Query:                        # stands in for PlannedQuery: the supervisor builds follow-ups with type(plan.queries[0])
    text: str
    filter: object = None


class Entities:
    topics = []

    def model_dump(self):
        return {}


def make_plan(q):
    return SimpleNamespace(queries=[Query(q)], planned=False, intent=SimpleNamespace(value="lookup"),
                           domain=SimpleNamespace(value="cit"), tax_year=2026, entities=Entities(), assumptions=[])

Q = "What is the corporate income tax rate in Singapore?"


class Retriever:
    mode = "dense"                      # no reranker: scores kept as they are

    def __init__(self, faults=()):
        self.faults, self.calls = list(faults), 0     # one entry per search() call: None | "empty" | "error"

    def search(self, q, k=3, min_score=None, **kw):
        self.calls += 1
        f = self.faults.pop(0) if self.faults else None
        if f == "error":
            raise RuntimeError("injected retrieval failure")
        if f == "empty":
            return []
        return [{"id": f"{q[:20]}-{i}", "text": f"chunk about {q} #{i}", "score": 0.9} for i in range(2)]


class Auditor:
    def __init__(self, sufficient=True):
        self.sufficient = sufficient

    def audit(self, question, chunks, kept_context="", tried=(), year=None):
        return S.RetrievalAudit(relevant=[True] * len(chunks), reasons=[""] * len(chunks),
                                sufficient=self.sufficient, missing=None if self.sufficient else "rebate")


class Store:
    def remember(self, *a): return []
    def save_pending(self, *a): pass
    def pop_pending(self, *a): return None


def make_llm(decides):
    decides = list(decides)
    seen = []

    def llm(messages, schema=None):
        enum = ((schema or {}).get("properties", {}).get("action", {}).get("enum")) or []
        if set(enum) == {"search", "decompose"}:
            return json.dumps({"action": "search", "reason": "one fact"})
        if "run_tool" in enum:
            seen.append(messages[1]["content"])
            return json.dumps(decides.pop(0))
        return "The rate is 17% [Doc 1]."
    llm.seen = seen
    return llm


def run(faults=(), decides=(), sufficient=True):
    ret = Retriever(faults)
    llm = make_llm(decides)
    sup = S.Supervisor(Store(), S.RetrievalAgent(ret), Auditor(sufficient), llm=llm, planner=lambda *a: [])
    out = sup.supervise("u", make_plan(Q), Q)
    return out, llm.seen, ret


def check(name, cond, out, seen):
    print(f"{'PASS' if cond else 'FAIL'}  {name}: status={out.status} stop={out.stop_reason}")
    if not cond:
        print("   trace:\n   " + "\n   ".join(out.trace))
        print("   decide prompts:", seen)
    return cond


ok = True
# 1. no fault: unchanged behaviour
out, seen, _ = run()
ok &= check("no fault", out.status == "answered" and out.stop_reason == "auditor_sufficient" and not seen, out, seen)

# 2. --fault error: the model sees the failure and retries the same query (allowed: it was never searched)
out, seen, _ = run(faults=["error"], decides=[{"action": "search", "queries": [Q], "reason": "retry"}])
ok &= check("error -> retry same query", out.status == "answered" and out.stop_reason == "auditor_sufficient"
            and "FAILED" in seen[0], out, seen)

# 3. --fault empty: exact repeat blocked, the model is told and rewords; a keyword-subset rewording is allowed
out, seen, _ = run(faults=["empty"], decides=[{"action": "search", "queries": [Q], "reason": "again"},
                                               {"action": "search", "queries": ["corporate tax rate"], "reason": "x"}])
ok &= check("empty -> exact repeat blocked, rewording runs", out.status == "answered"
            and "NO results" in seen[0] and "repeats one already searched" in seen[1], out, seen)

# 4. two errors in a row: stop with tool_error, nothing vetted -> no_answer, no crash
out, seen, _ = run(faults=["error", "error"], decides=[{"action": "search", "queries": [Q], "reason": "retry"}])
ok &= check("error twice -> tool_error", out.status == "no_answer" and out.stop_reason == "tool_error", out, seen)

# 5. give_up with vetted chunks -> no_answer (was: answered)
out, seen, _ = run(decides=[{"action": "give_up", "reason": "docs do not answer this"}], sufficient=False)
ok &= check("give_up respected", out.status == "no_answer" and out.stop_reason == "agent_gave_up", out, seen)

# 6. repeated queries: the model gets one more decision before no_new_queries ends the loop
out, seen, _ = run(decides=[{"action": "search", "queries": [Q], "reason": "again"},
                            {"action": "answer", "reason": "enough"}], sufficient=False)
ok &= check("repeat -> model decides again", out.status == "answered" and out.stop_reason == "agent_answered"
            and len(seen) == 2 and "repeats one already searched" in seen[1], out, seen)

# 7. repeated twice -> no_new_queries as before
out, seen, _ = run(decides=[{"action": "search", "queries": [Q], "reason": "again"},
                            {"action": "search", "queries": [Q], "reason": "again"}], sufficient=False)
ok &= check("repeat twice -> no_new_queries", out.stop_reason == "no_new_queries" and out.status == "answered", out, seen)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)