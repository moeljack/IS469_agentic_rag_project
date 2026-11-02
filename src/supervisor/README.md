# Supervisor and harness

The supervisor is the agent that answers a user's Singapore tax question. It takes the intake gate's result,
decides each next action (search, decompose, run a tool, ask the user, answer, give up), has every retrieved chunk
checked by the auditor, and writes the final cited answer from approved chunks only.

The harness (`harness.py`) wires the real components together and gives you a terminal chat.

```
src/supervisor/
  supervisor.py       the agent loop, plus the RetrievalAgent and Auditor adapters
  harness.py          builds the real stack and runs the terminal chat
  test_supervisor.py  offline tests: loop, adapters, user store, speed guards
  test_routing.py     offline tests: intake gate, route step, decompose step
```

---

## Quick start

**Prerequisites**

1. Python environment set up (see the main `README.md`).
2. Ollama running, with the model pulled: `ollama pull qwen3.5:4b`
3. The ChromaDB collection built: `python src/ingest/ingest.py` (and `ingest_web.py` for scraped pages).

**Run**, from the project root:

```powershell
python src/supervisor/harness.py --user acme            # chat
python src/supervisor/harness.py --user acme --trace    # chat, and print what each agent did
python src/supervisor/harness.py --user acme --forget   # delete everything stored for this user
```

Type a question at `you>`. Press Enter on an empty line to quit.

On start-up the harness loads the Ollama models and the reranker, so the first question doesn't pay for that
(about 25 s for the reranker). Skip this with `--no-warm`.

---

## Harness flags

| Flag | Default | Effect |
|---|---|---|
| `--user ID` | required | One profile per id. Saved company facts are reused across sessions. |
| `--trace` | off | Print the step-by-step trace and a timing line after each answer. |
| `--max-steps N` | 4 | Action budget per question. After this the supervisor answers with what it has. |
| `--k N` | 3 | Chunks kept per query after reranking. |
| `--max-chunks N` | 6 | Chunks handed to the auditor per search. |
| `--max-queries N` | 2 | Queries run per search (planned or follow-up). |
| `--candidates N` | 6 | Chunks the reranker scores per query. `0` = `CANDIDATES` in `src/config.py`. Lower is faster. |
| `--audit-min-score S` | 0.5 | Chunks the reranker scores below `S` are rejected without an auditor call. `0` = audit all. |
| `--quantize-rerank` | off | int8 reranker weights. Faster on CPU, slightly less accurate. Not for Apple MPS. |
| `--intake-plans` | off | Original pipeline: the intake call also writes the sub-queries and there is no route step. |
| `--no-warm` | off | Don't preload the models and reranker at start-up. |
| `--company-name`, `--uen` | | Save company details to the user's profile. |
| `--forget` | | Delete the user's profile, pending state and history, then exit. |

## Environment variables

The default models are set in `src/config.py`; the variables below override them.

| Variable | Default | Used for |
|---|---|---|
| `INTAKE_MODEL` | `qwen3.5:4b` | Intake extraction |
| `PLANNER_MODEL` | `INTAKE_MODEL` | Query planner (the `decompose` step) |
| `SUPERVISOR_MODEL` | `INTAKE_MODEL` | Route, decide and answer calls |
| `AUDITOR_MODEL` | `qwen3.5:4b` | Auditor |
| `OLLAMA_KEEP_ALIVE_CLIENT` | `30m` | How long Ollama keeps the models loaded after a call |
| `RERANK_QUANTIZE` | unset | `1` = int8 reranker (same as `--quantize-rerank`) |
| `INTAKE_DEBUG` | unset | Print the intake model's raw JSON |

Keep every model the same unless you have the memory for several. With one shared model Ollama never has to swap.

---

## How a turn runs

```
user message
   |
   v
UserStore ------------- saved company facts + pending clarification -> `prior`
   |
   v
Intake gate (decomposer.intake_and_plan)
   extract():  one LLM call -> domain, intent, entities          (no search queries)
   resolve():  deterministic gates
       out of scope            -> Refuse   -> reply, stop
       mandatory fact missing  -> Clarify  -> ask, save pending, stop
       otherwise               -> Decompose(planned=False, queries=[the question])
   |
   v
Supervisor loop (max_steps actions)
   step 1   route:  "search" as written  |  "decompose" via the query planner
   step 2+  decide: search | decompose | run_tool | ask_user | answer | give_up
            every search: retrieve -> reranker cutoff -> auditor -> keep approved chunks
   |
   v
Answer: written from approved chunks only, cited [Doc n], with sources and assumptions
```

### 1. Profile and prior

`handle_turn()` loads (or creates) the user's profile. Saved company facts (`fye`, `is_new_company`,
`incorporation_year`, `tax_resident`) are merged with any entities saved from a clarification asked last turn.
The result is the intake's `prior`, so the user doesn't have to repeat themselves.

### 2. Intake gate

The intake only classifies the request and extracts facts. It writes no search queries. Its job is to stop requests
that should never reach retrieval:

- **Refuse**: domain or intent is `out_of_scope`.
- **Clarify**: a fact required for this kind of question is missing.

| Domain | Intent | Required facts |
|---|---|---|
| CIT | computation | `chargeable_income` |
| CIT | deadline | `fye` |
| CIT | eligibility (SUTE) | `is_new_company` or `incorporation_year` |
| Property tax | computation | `annual_value`, `is_owner_occupied` |
| GST | computation | `taxable_turnover` |
| Stamp duty | computation | `purchase_price_or_market_value`, `buyer_profile` |

It also normalises facts, e.g. it checks dates are real and derives the YA from the financial year end
(FYE in 2025 → YA 2026).

### 3. The supervisor loop

**Step 1, route.** The supervisor decides how to search before anything has been searched:

- `search`: the question asks for one fact, rate, rule, deadline or procedure. It is searched as written.
- `decompose`: answering needs several separate facts (several conditions, a rate plus a rebate, a comparison,
  a computation). The query planner writes one keyword query per fact, and those are searched.

If the route output is unusable, or the planner returns nothing, the question is searched as written.
With `--intake-plans`, the intake has already written the queries, so step 1 searches them and there is no route call.

**Steps 2 and later, decide.** The supervisor sees the question, what has been vetted so far, the log of earlier
actions (including what the auditor said is missing) and picks one action:

| Action | What happens |
|---|---|
| `search` | Runs 1–3 new keyword queries the model writes. If they all repeat earlier ones, the auditor's suggestions are used instead. |
| `decompose` | The planner writes queries only for the fact the auditor reports as missing, and they are searched in the same step. |
| `run_tool` | Runs a registered tool (e.g. `compute_tax`). Its output becomes a citable document. Offered only for computation questions. |
| `ask_user` | Ends the turn with a question. Entities are saved so the next turn resumes. |
| `answer` | Write the answer from the vetted chunks. |
| `give_up` | Reply that the sources don't cover this. |

**Every search** goes through the same steps:

1. **Retrieve.** `RetrievalAgent` runs each query through `Retriever.search()` (vector + BM25 + reranker), drops
   chunks already judged, merges, sorts by reranker score and keeps the top `max_chunks`.
2. **Reranker cutoff.** Chunks scoring below `audit_min_score` are rejected without an auditor call, except the best
   `audit_keep_top` (3) of each search, which are always audited. A chunk's score is the better of its score for the
   question and for the query that found it, so the chunk answering one part of a multi-part question isn't dropped.
3. **Audit.** `Auditor` asks the auditor model whether each chunk is related to the question. It also reports
   whether the approved chunks are sufficient, what is missing, and a suggested query. If the auditor fails, chunks
   are kept unjudged and the trace says so.
4. **Keep.** Only approved chunks are ever shown to the answer model.

The auditor and the answer model see each chunk's text up to `AUDIT_CHARS` / `DOC_CHARS` (2,000 characters each).
A longer chunk is not simply cut: `excerpt()` shows its opening plus the part that best matches the question.

### 4. Guard rails

A 4B model isn't trusted to run the loop unchecked, so code enforces these rules:

- Step 1 is always a search.
- `answer` with nothing vetted becomes `give_up`. The supervisor never answers from nothing.
- Unparseable or invalid model output falls back to `answer` if something is vetted, otherwise `give_up`.
- Follow-up queries that repeat or nearly repeat an earlier query are dropped before retrieval. Near-repeat means
  same keywords after removing stopwords, plurals, bare numbers and synonyms like requirement/condition/criteria.
- When the auditor reports the approved chunks are sufficient, the loop answers without another decide call.
- When every auditor suggestion repeats an earlier query, the loop answers without another decide call.
- A follow-up search that returns no chunks it hasn't seen before ends the loop.
- Two searches in a row that approve nothing, with nothing vetted, end the loop.

These show up as the result's `stop_reason`:

| `stop_reason` | Meaning |
|---|---|
| `auditor_sufficient` | The auditor said the approved chunks answer the question |
| `agent_answered` | The model chose `answer`, or a search found nothing new after something was vetted |
| `suggestions_repeat` | Every auditor suggestion repeated an earlier query |
| `no_new_queries` | The model's and the auditor's queries were all repeats |
| `max_steps` | The action budget ran out |
| `no_progress` | Two searches in a row approved nothing |
| `agent_gave_up` | The model chose `give_up`, or there was nothing to answer from |
| `agent_asked_user` | The model asked the user a question |

### 5. Answer

The answer model gets the question, the user's facts, any assumptions, and up to 6 vetted chunks (1,200
characters each) labelled `[Doc 1]`, `[Doc 2]`, and so on. It must cite every factual claim and may not state a rate,
cap, date or amount that isn't in the documents.

- If the loop stopped early (`max_steps`, `no_new_queries`) or the last audit wasn't fully satisfied, the prompt
  includes a **coverage note**, and the answer says which part isn't covered.
- If the model replies `INSUFFICIENT_CONTEXT`, the user gets the "sources don't cover this" message instead.
- The reply ends with **Assumptions** (e.g. a defaulted YA, or saved company details that were reused) and
  **Sources** (the sources the answer cited).

### 6. After the turn

- Company facts from this turn are saved to the profile. Per-question facts (chargeable income, the YA asked about)
  are only written to the history log.
- After a `clarify` reply the harness keeps the conversation, so the next message is read as the answer. After any
  other reply it starts a fresh conversation. The profile persists either way.

---

## Reading `--trace`

The first line shows the result's status and the route step 1 took (`planned`, `search` or `decompose`).
An example for a SUTE eligibility question (illustrative):

```
    [answered route=decompose] profile loaded for acme; saved facts: {'is_new_company': True}
    profile updates: none
    step 1: decompose - needs several separate qualifying conditions
      planner queries: ['start-up tax exemption qualifying conditions', 'start-up tax exemption shareholders']
      retrieved 3 chunk(s), 3 new, for 'start-up tax exemption qualifying conditions'
      retrieved 3 chunk(s), 1 new, for 'start-up tax exemption shareholders'
      search [...]: 2/3 chunks related, 1 below reranker cutoff; auditor suggests: nothing
        approved [0.98] Start-Up Tax Exemption > Qualifying conditions
        approved [0.91] Start-Up Tax Exemption > Shareholding
        REJECTED [0.62] Partial Tax Exemption: about PTE, not SUTE
        skipped [0.31] Form C-S: reranker score below 0.50, not audited
      guard: auditor reports the approved chunks are sufficient; answering.
    answered from 2 document(s)
    timing: intake 2.4s | route 0.9s | plan 1.6s | retrieval 8.1s | audit 4.2s | answer 5.3s | other 0.4s | total 22.9s
```

How to read it:

| Line | Meaning |
|---|---|
| `step N: <action> - <reason>` | What the supervisor chose and why |
| `retrieved ... for '<query>'` | One retrieval per query, and how many chunks were new |
| `approved` / `REJECTED [score]` | The auditor's verdict on each chunk, with the reranker score |
| `skipped [score]` | Rejected by the reranker cutoff, never sent to the auditor |
| `guard: ...` | A code rule stopped or shortened the loop |
| `timing:` | Seconds per stage. `(x2)` = the stage ran twice. `other` = profile store and glue code |

---

## Using it from Python

```python
from src.supervisor.harness import build_supervisor

sup = build_supervisor()                       # same defaults as the harness
out = sup.handle_turn("acme", [{"role": "user", "content": "What is the corporate income tax rate?"}])

out.status        # "answered" | "clarify" | "refused" | "no_answer"
out.message       # what to show the user (answer + assumptions + sources)
out.route         # "planned" | "search" | "decompose"
out.stop_reason   # see the table above
out.sources       # cited source URLs / PDF names
out.trace         # the --trace lines
out.timings       # [(stage, seconds), ...]
out.audits        # one RetrievalAudit per audited search
```

To add a calculator, pass it as a tool. It gets the `Decompose` plan and returns text:

```python
sup = build_supervisor(tools={"compute_tax": lambda plan: f"Tax payable: ..."})
```

Tools are only offered to the model for computation questions. The harness registers none by default.

`Supervisor(...)` also takes injectable `llm`, `intake`, `planner` and an `Auditor(judge=...)`, which is how the tests
run without Ollama.

---

## Comparing the two pipelines

| | Default | `--intake-plans` |
|---|---|---|
| Intake call | Extraction only | Extraction and sub-queries |
| Who decides to decompose | Supervisor route step | Always decomposed by the intake |
| Simple question | Intake + route call | Intake (larger output) |
| Complex question | Intake + route + planner calls | Intake (larger output) |
| Re-planning mid-loop | `decompose` for the missing fact | Model's own queries / auditor suggestions only |

Run your eval questions both ways and compare accuracy, total LLM calls, latency and the `route` / `stop_reason`
counts.

---

## Tests

No Ollama, models or vector database needed (ChromaDB runs in memory):

```powershell
python -m pytest src/supervisor/ -q                       # everything
python -m pytest src/supervisor/test_routing.py -v        # intake gate, route and decompose only
python -m pytest src/supervisor/ -k "route" -v            # tests whose name matches
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `... did not stay loaded` at start-up | Ollama evicted a model when the next one loaded. Use one model for everything (e.g. `AUDITOR_MODEL=qwen3.5:4b`) or check `ollama ps`. |
| Trace says `auditor unavailable, kept unjudged: ...` | The auditor model failed (often "model not found"). Chunks were kept unchecked. Pull the model or set `AUDITOR_MODEL`. |
| `route unparseable` in the trace | The route call returned bad JSON. The question was searched as written, so the turn still works. |
| Slow answers | Retrieval is usually the largest stage. Try `--candidates 4`, `--quantize-rerank` (CPU only) or `--max-queries 1`. Read the `timing:` line to see which stage dominates. |
| Every answer is "sources don't cover this" | Check the collection was ingested (`python src/ingest/inspect_db.py`), and lower `--audit-min-score` to see whether the cutoff is rejecting everything. |
| `ImportError: IntakeOutput` or `ROUTE_SCHEMA` | `decomposer.py`, `decomposer_general_schemas.py` and `supervisor.py` are from different versions. Update all three. |