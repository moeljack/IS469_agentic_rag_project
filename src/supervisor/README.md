# Supervisor

The supervisor is the agent that answers a user's Singapore tax question. It takes the intake gate's result, picks
each next action, has the auditor check every retrieved chunk, and writes the final cited answer from approved chunks
only. `harness.py` builds the real components and runs a terminal chat.

```
supervisor.py       the agent loop, plus the RetrievalAgent and Auditor adapters
harness.py          builds the real stack, terminal chat
test_supervisor.py  offline tests: loop, adapters, user store
test_routing.py     offline tests: intake gate, route step, decompose step
```

## Run

You need Ollama running with `qwen3.5:4b` pulled, and a built ChromaDB collection (see the main README).

```powershell
python src/supervisor/harness.py --user acme            # chat
python src/supervisor/harness.py --user acme --trace    # also print each step and its timing
python src/supervisor/harness.py --user acme --forget   # delete this user's stored data
```

The flags you'll use most:

| Flag | Default | Effect |
|---|---|---|
| `--max-steps` | 4 | Action budget per question |
| `--k` / `--max-chunks` / `--max-queries` | 3 / 6 / 2 | Chunks kept per query / chunks audited per search / queries per search |
| `--audit-min-score` | 0.5 | Chunks the reranker scores below this skip the auditor (`0` = audit all) |
| `--intake-plans` | off | Original pipeline: the intake writes the sub-queries and there is no route step |

The models are set in `src/config.py` and can be overridden with `SUPERVISOR_MODEL`, `PLANNER_MODEL` and
`AUDITOR_MODEL`. Use one model for all of them unless you have the memory for several.

## How a turn runs

1. **Profile.** Saved company facts and any pending clarification are loaded as the intake's `prior`.
2. **Intake gate.** The intake classifies the request and extracts entities. It **refuses** out-of-scope requests and
   **clarifies** when a required fact is missing (e.g. chargeable income for a computation). Otherwise the question
   goes to the loop.
3. **Step 1, route.** `search` the question as written (one fact), or `decompose` it with the query planner (several
   facts, a comparison or a computation).
4. **Steps 2 onward, decide.** The model sees what has been vetted, plus the log of earlier actions and what the
   auditor said is missing, and picks one action:

   | Action | Effect |
   |---|---|
   | `search` | 1-3 new keyword queries, written by the model |
   | `decompose` | The planner writes queries only for the fact the auditor reports missing |
   | `run_tool` | Runs a registered calculator; offered only for computation questions |
   | `ask_user` | Ends the turn with a question; the entities are saved so the next turn resumes |
   | `answer` / `give_up` | Answer from the vetted chunks, or reply that the sources don't cover it |

5. **Every search:** retrieve (hybrid + rerank) → drop chunks below the reranker cutoff (the best 3 per search are
   always audited) → the **auditor** approves or rejects each chunk and reports whether the approved set is
   sufficient → keep only the approved chunks.
6. **Answer.** The model gets up to 8 vetted chunks labelled `[Doc n]`. It must cite every claim and may not state a
   rate, date or amount that isn't in them. If the loop ended early, a coverage note makes the answer say which part
   isn't covered. The reply ends with **Assumptions** and **Sources**.

## Guard rails

A 4B model isn't trusted to run the loop unchecked, so these rules are enforced in code:

- Step 1 is always a search. If the route or planner output is unusable, the question is searched as written.
- `answer` with nothing vetted becomes `give_up`. Unparseable output falls back to `answer` if anything is vetted,
  otherwise `give_up`.
- Queries that repeat or nearly repeat an earlier search (same keywords once stopwords, plurals and synonyms are
  removed) are dropped before retrieval.
- When the auditor says the approved chunks are sufficient, the loop answers without another decide call.
- After an error (a failed search, every query a repeat, or zero new chunks), the model gets one more decide call
  before the loop stops.
- If the auditor fails, chunks are kept unjudged and the trace says so.

Why the loop stopped is recorded as `stop_reason`: `auditor_sufficient`, `agent_answered`, `suggestions_repeat`,
`no_new_queries`, `max_steps`, `no_progress`, `tool_error`, `agent_gave_up` or `agent_asked_user`.

## From Python

```python
from src.supervisor.harness import build_supervisor

sup = build_supervisor(tools={"compute_tax": lambda plan: "Tax payable: ..."})   # tools are optional
out = sup.handle_turn("acme", [{"role": "user", "content": "What is the corporate income tax rate?"}])
out.status, out.message, out.route, out.stop_reason, out.sources, out.trace
```

`Supervisor(...)` also accepts injected `llm`, `intake`, `planner` and `Auditor(judge=...)` objects, which is how the
tests run without Ollama:

```powershell
python -m pytest src/supervisor/ -q
```