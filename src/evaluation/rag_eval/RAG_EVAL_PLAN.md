# IS469 RAG Evaluation Plan

Oct 28, 2026 · @Joel · updated 5 Nov 2026 (checked against the code at commit `9df16fd`)

## Status (5 Nov 2026)

**Where things stand**

| Item | State |
| --- | --- |
| Dataset | 80 records: 60 answerable (S01–S21 without S20, M01–M20, A01–A10, C01–C10) + 20 out-of-scope (O01–O20). Gold labels regenerated and reviewed; 45 records still carry `needs_review` (Section 2.3) |
| Stage 1 retrieval (R0–R4) | Run on the 51-question set. **Rerun needed** for the current set: `run_all.py` refuses to select a mode until the retrieval files cover all 60 answerable questions |
| Stage 1–2 answers (R0, N1, N2, A1) | Done on the cluster GPU for the earlier set. **To resume:** S10 (reworded) and C01–C10 (new). Delete S10's rows and the M02/M12 judge rows first (Section 9.3) |
| Judge | `gemma4:12b` via Ollama on the cluster GPU (`configs/judge.yaml`). It passed `check_judge.py`. **The 30-pair human validation (Section 4.3) is not done yet**; there are no labels or kappa script in the repo |
| Stage 3 (API generators) | Not run. Check the config mismatch in Section 5, Stage 3 before running |
| Failure coding, qualitative cases | Not started |

**Provisional results** (the 50 primary questions; not final until the reruns above are done)

| Config | Part recall@5 | P@5 | Hit@5 | MRR | Median retrieval latency (cluster GPU) |
| --- | --- | --- | --- | --- | --- |
| R0 dense | 74.7% | 0.30 | 60% | 0.81 | 57 ms |
| R1 BM25 | 71.3% | 0.26 | 60% | 0.68 | 3 ms |
| R2 hybrid | 77.0% | 0.31 | 62% | 0.80 | 62 ms |
| R3 dense + rerank | 88.3% | 0.37 | 80% | 0.90 | 653 ms |
| R4 hybrid + rerank | 85.3% | 0.35 | 76% | 0.88 | 668 ms |

These are the saved Stage 1 rankings rescored against the current gold labels, for the 50 primary questions.

- **The reranker gives almost all of the gain:** about +11 to +14 points of recall and +16 to +20 points of Hit@5.
- **Hybrid adds little:** about +2 points without the reranker, and −3 points with it.
- **The rule still selects R4:** its recall is not below R0's (Section 5, Stage 1), although R3 is slightly higher. Report that as a finding; a 3-point gap is about 1–2 facts.

| Run | Correct (score 2) on 49 primary questions | Note |
| --- | --- | --- |
| R0 naive (s1_naive_r0_dense) | 32 | |
| N1 naive, R4, k = 5 | 37 | |
| N2 naive, R4, k = 6 | 38 | k = ceil(A1's mean 5.26 chunks in the answer prompt) |
| A1 agentic | 42 | Answered all 49 (no clarify or refusal). Median latency about 35 s against about 10 s for N1 |

S10 is missing from these 49 (it was reworded). The M02 and M12 judge scores predate their reference fixes. No CIs yet.

**Changes since the 31 Oct draft**

- **5 Nov: intake-safe wording.** The real intake gate was run on all 60 answerable questions. 4 would have drawn a clarifying question (S10, C03, C04, C06) and were reworded to state the fact the gate requires (Section 2.3). All 60 now pass the gate.
- **5 Nov: calculation questions.** C01–C10 added (6 single-rule, 4 multi-step), and S20 dropped, since it repeated A06 part 1. They are reported as a separate group, outside the headline (Section 2.3).
- **4–5 Nov: gold review.** 17 question parts gained gold chunks, and some references were corrected (M02, M12) and O17 replaced. Gold is regenerated with `label_gold.py --write`; M09's hand-added part-1 chunk `web:e045f8905e:1` must be re-added after every run.
- **3 Nov: paraphrases out, out-of-scope up to 20.** P01–P05 moved to the agentic team's robustness suite (`agent_eval/data/robustness_cases.json`). Their old results are in `results_stale/retrieval_paraphrases/`. O05–O20 were added, tagged `oos_group`.
- **Earlier:** Question ids changed from `Q01`–`Q64` to S/M/A/C/O; `qid_map.json` and `db_check/apply_qid_map.py` convert old results. RAGAS was replaced by the two-call judge scorer `score_grounding.py`. The answer step's chunk cut went from 1,200 to 2,000 characters, with an excerpt matched to the question. The Stage 3 "OpenAI" generator is gpt-oss-120b on Groq. Model names now live in `src/config.py`.

## Summary and key decisions

Evaluate the RAG system in three stages, plus a scoring stage:

1. A retrieval ablation.
2. Naive vs agentic RAG.
3. A generator LLM comparison.

Each stage carries its winner forward, so the team runs about 8 generation runs instead of a full grid. Stage 2 includes a naive baseline given the same amount of context as the agent, so agentic RAG isn't credited merely for seeing more chunks.

**One primary metric, accuracy (correctness = 2) on the 50 primary questions, carries the headline claims.** All other metrics are diagnostic. Calculation questions and out-of-scope questions are reported as their own groups (Section 2.3).

**Is the dataset big enough? Just enough, with caveats.** One primary question moves accuracy by 2 points (1 in 50). Per-question paired comparisons (Section 7) are the main evidence; averages alone are not.

**Gold passages are auto-labelled from verbatim facts, then audited** (Section 2.4). The corpus is not re-ingested, so chunk ids stay valid.

**What "agentic RAG" means here** is the whole supervisor stack described in Section 0, not retrieval as a single tool. The report must say so.

## 0. System architecture

The chatbot answers Singapore corporate income tax (CIT) questions from IRAS material. Every LLM in the system is `qwen3.5:4b` run through Ollama, at temperature 0 with thinking off (`src/config.py`: `INTAKE_MODEL`, `PLANNER_MODEL`, `SUPERVISOR_MODEL`, `AUDITOR_MODEL`).

### 0.1 Corpus and index

| Part | Detail | Code |
| --- | --- | --- |
| Sources | 56 IRAS corporate-tax web pages, scraped 14 Oct 2026 (`data/raw/iras_corporate_tax.jsonl`). Two IRAS e-Tax guide PDFs (`data/manual/`): the REIT ETF guide and the general insurers' IBNR reserves guide | `src/ingest/ingest_web.py`, `ingest.py` |
| Chunking | Web pages: section-aware chunks, with headings recovered from Markdown or Title-Case lines. PDFs: Docling `HybridChunker`, with tables kept as Markdown rows and two heading fixes. At most 512 tokens per chunk (`MAX_TOKENS`) | `web_chunking.py`, `chunking.py` |
| Store | One ChromaDB collection, `manuals` (546 chunks), in the git-tracked `chroma_db/`. Each chunk has `source`, `headings` (the heading path), `pages`, `source_type` and `title`. Ids are positional: `web:<10 hex>:<n>` or `<pdf name>:<n>` | `src/ingest/store.py` |
| Embedder | `Qwen/Qwen3-Embedding-0.6B`, with the query prompt for queries; normalised, cosine similarity | `src/config.py` |

### 0.2 Retrieval (`src/retrieval/retriever.py`)

`Retriever.search(query, k, candidates=10)` runs three stages:

1. **Candidates.** Vector search and BM25 each return their top `candidates` (10) chunks. BM25 runs over heading path + text, so exact terms such as "section 45G" or "Annex 2B" match. The two lists are merged by reciprocal rank fusion (constant 60).
2. **Rerank.** The cross-encoder `BAAI/bge-reranker-v2-m3` scores each candidate against the query. It reads heading path + text, up to 1,024 tokens, and gives a 0–1 relevance.
3. **Top k** by reranker score.

The evaluation's `ModeRetriever` (`rag_eval_retrieval/retrieval_modes.py`) builds R0–R3 from these stage methods. R4 is `search()` unchanged.

### 0.3 The agent (`src/supervisor/`, `src/decomposer/`, `src/auditor/`)

```text
question
  |
  v
UserStore profile -> saved company facts become the intake's "prior"
  |
  v
Intake gate  decomposer.intake_and_plan: 1 LLM call extracts domain, intent, entities; resolve() then applies rules
  |- domain or intent out_of_scope ........................................ REFUSE  (status refused)
  |- a mandatory fact is missing AND the question contains a digit ........ CLARIFY (status clarify)
  |     (CIT computation needs chargeable_income; CIT deadline needs fye; SUTE eligibility needs the incorporation year)
  |- otherwise: one query = the question (a missing fact without digits only adds an "answered in general terms" note)
  v
Supervisor loop, at most max_steps = 4 actions
  step 1   route: "search" the question as written, or "decompose": the planner writes one query per fact
  step 2+  decide: search | decompose | run_tool | ask_user | answer | give_up
  every search:
     RetrievalAgent: Retriever.search() per query (k each) -> drop chunks already judged -> rescore against the
                     question with the reranker -> keep the best max_chunks
     reranker cutoff: below audit_min_score (0.5) is rejected unaudited, except the top 3, which are always audited
     Auditor LLM:    approves or rejects each chunk; says whether the approved set suffices, what is missing, and a
                     follow-up query
  guard rails in code, e.g. answer at once when the auditor says "sufficient"; stop on repeated queries or no progress
  v
Answer  ANSWER_SYSTEM prompt; up to 8 approved chunks (MAX_DOCS), each shown as a 2,000-character excerpt that best
        matches the question (DOC_CHARS, supervisor.excerpt); cite [Doc n]; no number that is not in the documents
  |- reply starts with INSUFFICIENT_CONTEXT, or nothing approved ............... status no_answer
  |- otherwise .............................................................. status answered (+ assumptions, sources)
```

- **Statuses.** `handle_turn()` returns one of `answered`, `clarify`, `refused` or `no_answer`, with a `stop_reason` such as `auditor_sufficient`, `max_steps` or `no_progress`.
- **The gate is code, not the LLM.** Refuse and clarify decisions at intake are deterministic, made on what the intake call extracted. The supervisor can also choose `ask_user` later. It never did in the earlier A1 run.
- **Tools.** A tool such as a tax calculator is offered only for computation questions. The evaluation registers none.

### 0.4 The naive baseline

One `search()` call returns the top k chunks; then one answer call. That call uses the same `ANSWER_SYSTEM` prompt, `[Doc n]` layout, 2,000-character excerpts and settings as the agent, with `USER FACTS: none` (`rag_eval_approach/answer.py`). A reply that is empty or starts with `INSUFFICIENT_CONTEXT` is a `no_answer`. So naive RAG can decline too.

### 0.5 How the evaluation runs each system

The code is `rag_eval_approach/systems.py`, driven by `run.py`.

| | Naive (`NaiveSystem`) | Agentic (`AgenticSystem`) |
| --- | --- | --- |
| Built from | `ModeRetriever(mode)` + `make_llm(provider, model)` | `harness.build_supervisor(...)` with the eval config's limits, `tools=None`, a temporary `UserStore` and the same `ModeRetriever` |
| Per question | One search, then one answer | `handle_turn("eval-<run>-<qid>", [question], today=2026-10-28)`: a fresh user each question and a fixed date |
| Logged | Top-k chunks (`retrieved_seen` = `context_used`), latency, tokens | Retrieval, the auditor and the answer prompt are wrapped. They log every chunk seen, the chunks put in the prompt, a per-search log (queries, new ids, audited and approved ids), per-stage latency and tokens summed over every Ollama call |
| What a generator swap changes (Stage 3) | The answer call | Only the supervisor's `llm` (route, decide, answer). The intake, planner and auditor stay on local Qwen |

The eval's agent limits differ from the CLI harness defaults: k = 5 per query (harness default 3), pool 10 (6), and `max_chunks` 8 (6). They match the naive systems' k and pool.

## 1. Objectives and research questions

The RAG evaluation answers four questions. Each maps to one stage and one table in the report.

| ID | Research question | Stage | Primary metric |
| --- | --- | --- | --- |
| RQ1 | Which retrieval configuration finds the right IRAS passages most reliably? | 1 | Part recall@5, MRR |
| RQ2 | Do hybrid search and reranking improve final answers, not just retrieval scores? | 1 | Accuracy, faithfulness |
| RQ3 | Does agentic RAG give better answers than naive RAG, including a naive baseline given the same amount of context, and is the gain worth the extra latency and tokens? | 2 | Accuracy, latency |
| RQ4 | How much does the supervisor/generator LLM matter, and does agentic RAG help a small model (Qwen 3.5 4B) more than larger ones? | 3 | Accuracy, faithfulness |

A secondary objective is to find *where* each system fails, by question type and source, through the qualitative analysis (Section 6).

## 2. Dataset preparation

The dataset is `src/evaluation/eval_questions.json`: 80 records, written with CRLF line endings, `indent=1` and `ensure_ascii=False`. `db_check/check_dataset.py` validates it against the corpus and writes nothing. Freeze the dataset before the final comparison runs.

### 2.1 Record format

Each record is one question. `facts` is a list of parts: each part is one fact the answer needs, written as one or more alternative verbatim strings from the corpus. `gold_passages` is filled by `label_gold.py` with one list of chunk ids per part.

```json
{"qid": "M02", "type": "multi", "tier": "webpage",
 "qtype": "multi-part", "source": "online_guide", "difficulty": "hard", "numeric": true,
 "answerable": true,
 "question": "What penalty applies if my company pays its tax after the due date, and what is the minimum monthly GIRO amount if it spreads payment under the instalment plan?",
 "reference_answer": "...",
 "key_facts": ["5%", "$50"],
 "facts": [["A 5% late payment penalty will be imposed on the unpaid tax"],
           ["minimum monthly GIRO deduction amount of $50"]],
 "gold_passages": [["web:<id>:<n>"], ["web:<id>:<n>"]],
 "needs_review": false}
```

- `key_facts` lists the numbers, dates or terms a correct answer must contain. The judge uses it for the numeric check.
- Out-of-scope records have `answerable: false`, empty `facts` and `gold_passages`, and an `oos_group`.

### 2.2 Tags

| Field | Values | Rule |
| --- | --- | --- |
| type | simple, multi | `simple` = exactly 1 fact; `multi` = 2+ facts, not all in one chunk (enforced by `check_dataset.py`) |
| qtype | definition, number, procedure, edge-case, multi-part, multi-step, computation, out-of-scope | `number` = a single fact that hinges on a rate, threshold, date or amount. `multi-part` = 2 independent facts. `multi-step` = 2–3 facts where one fact or condition leads to the next. `computation` = apply retrieved rules to the user's figures |
| source | online\_guide, reit\_etf\_guide, insurer\_guide, none | Where the answer lives |
| tier | webpage, pdf | Document type of the source |
| difficulty | easy, hard | `hard` = multi-fact, reasoning across sections, or a tricky condition |
| numeric | true, false | `true` if the answer hinges on a number, rate, threshold or date. Drives numeric accuracy |
| oos\_group | off-topic, near-miss, not-in-corpus | Out-of-scope records only (Section 2.3) |

### 2.3 The current test data

**Composition by group**

| Group | Ids | n | Analysis group | Notes |
| --- | --- | --- | --- | --- |
| Single-fact | S01–S21 (no S20) | 20 | Primary | definition, number, procedure and edge-case: 5 each |
| Multi-part | M01–M20 | 20 | Primary | 2 facts each |
| Multi-step | A01–A10 | 10 | Primary | 2–3 facts (6 with 3 facts) |
| Calculation | C01–C10 | 10 | Secondary (qtype `computation`) | 6 single-rule (C01–C06), 4 multi-step (C07–C10). The answer LLM does the arithmetic (no tools) and the judge needs exact figures |
| Out-of-scope | O01–O20 | 20 | Separate (Table 2e) | Refusal check only: off-topic 6, near-miss 8, not-in-corpus 6 |
| **Total** | | **80** | | 60 answerable |

**The 50 primary questions by tag**

| Tag | Breakdown |
| --- | --- |
| Question type | multi-part 20, multi-step 10, definition 5, number 5, procedure 5, edge-case 5 |
| Source | online guide 42, REIT ETF guide 4, insurer guide 4 |
| Tier | web page 42, PDF 8 |
| Difficulty | hard 36, easy 14 |
| Numeric flag | true 31, false 19 |
| Facts | 20 single-fact, 24 with 2 facts, 6 with 3 facts |

**Why three analysis groups** (`results_io.is_core`, `SECONDARY_QTYPES`):

- **Primary (50).** The answerable lookup questions; every headline number uses these.
- **Calculation (10).** Reported on their own, in Table 1a's extra row per mode, Table 2b's own row and the `accuracy_computation` column. They test applying a rule to figures, a different skill, and 10 questions are too few to move or support the headline. Their gold chunks hold the rule, not the result.
- **Out-of-scope (20).** Kept out of accuracy (below).
- **All 60 answerable together** are reported once as a robustness check (`accuracy_all_answerable`, the `all` group in `--pair`).

**Intake-safe wording (5 Nov).** The research questions compare answer quality, so every answerable question must reach retrieval and answering. A clarifying question would score 0 and leave nothing to compare. The agent's intake gate clarifies when a mandatory fact is missing and the question contains a digit (Section 0.3). So the questions state the facts the gate needs; for example, a calculation question gives the chargeable income.

- A one-off probe ran the real `intake_and_plan` (qwen3.5:4b, today = 2026-10-28, no prior) on all 60. S10, C03, C04 and C06 would have clarified; their rewordings pass, and none of the 60 is refused.
- Remaining risk: the supervisor's own `ask_user` action is an LLM choice and was not probed. It never occurred in the earlier A1 run. `clarified_answerable` and `clarified_computation` in `runs_summary` will show any cases.
- **Say this in the report's method:** "Questions state the facts the intake gate requires (e.g. chargeable income), so every question reaches retrieval and answering; clarification behaviour is evaluated in the agentic team's suites."

**The 20 out-of-scope questions** (`oos_group`) test negative rejection: does a system decline a question the corpus cannot answer?

- **off-topic** (O05–O10): another country's tax, CPF, ACRA incorporation, work passes, a US filing, bank loans.
- **near-miss** (O01, O03, O11–O16): GST, stamp duty, property tax, personal income tax and the foreign worker levy. The corpus mentions these terms but not the rate or rule asked for. Retrieval returns plausible chunks, and the system has to notice they don't answer.
- **not-in-corpus** (O02, O04, O17–O20): in-domain CIT questions about a year, figure or scheme the documents do not hold, e.g. Pillar Two, a YA 2027 rebate, the YA 2008 rate, tax collections, Budget 2027 and VCCs.

**How out-of-scope results are reported (RQ3, RQ4).**

- `score_judge.py` gives each out-of-scope answer an `outcome`:
  - declined: status refused or no_answer, or an `INSUFFICIENT_CONTEXT` reply;
  - clarified: asked a question back;
  - answered: a hallucination risk;
  - error, or unscored.
- `analyze.py` writes Table 2e per run and per `oos_group`. It shows the decline rate with a Wilson 95% CI next to the answerable questions each system wrongly declined or clarified, since a system that declines everything would otherwise look perfect.
- In RQ3, say *where* each system declines: the agent's intake gate before retrieval, or the answer model after it.
- In RQ4, add the decline rate as a column per generator.
- At n = 20 only a large gap (e.g. 6/20 vs 14/20) is significant.
- Gate-level testing (adversarial wording, intake classifier errors) belongs to the agentic team's safety and robustness suites.

**Review status.** 45 records still carry `needs_review`: S 4, M 5, A 10, C 10, O 16. The C reference figures were recomputed in Python from the rule parameters, and the rule text checked against the gold chunks (Section 2.4). A teammate should still check the flagged references and key facts before the final scoring.

### 2.4 Labelling gold passages

`db_check/label_gold.py` scans every chunk in ChromaDB. For each part of each question, it records the ids of the chunks whose heading path + text contains any of that part's alternative strings. Matching ignores case, whitespace and apostrophe style (`db_check/facts.py: norm`). Facts that match no chunk, or more than 5, are flagged for a hand fix.

1. Run `python src/evaluation/rag_eval/db_check/label_gold.py` (a report only), fix flagged facts, then run it again with `--write`.
2. Re-add M09's hand-added gold chunk (`web:e045f8905e:1`, part 1); `--write` drops it.
3. Run `check_dataset.py`. Then `git restore -- chroma_db`, because opening the collection rewrites files in the tracked `chroma_db/`.
4. Audit: open the matching chunks of a sample of questions (at least 3 per source, mostly multi-fact). Confirm each holds the information, not just a heading or a generic phrase. The 4–5 Nov review covered all answerable questions and added gold to 17 parts.

Out-of-scope questions keep empty gold lists. Chunk ids are positional, so re-run the labelling after any re-ingest.

## 3. Metrics

There are four groups of metrics:

- retrieval, which needs no LLM;
- faithfulness and context precision, graded by the judge;
- a correctness rubric;
- efficiency.

BLEU and ROUGE are skipped, because they reward word overlap rather than correctness.

**The primary metric is accuracy: the % of primary questions with correctness = 2** (Section 3.3). Every headline claim about which system or configuration is better rests on this one number. Everything else is diagnostic: it explains *why* accuracy differs, and is not used to declare a winner.

### 3.1 Retrieval metrics (vs gold passages)

These are computed on the top k chunks (k = 5) for answerable questions only (`rag_eval_retrieval/metrics.py`). Gold is stored as **parts**: a part is covered if any of its chunk ids is retrieved, so a duplicate or equivalent chunk is not penalised.

```text
Part recall@k = parts covered / parts
Hit@k         = 1 if every part is covered
Precision@k   = gold chunks among the top k / k
F1@k          = 2 * P * R / (P + R)
MRR           = 1 / rank of the first gold chunk, over the full retrieved list (depth 10 in Stage 1)
```

| Metric | What it tells you | Note |
| --- | --- | --- |
| Part recall@5 | Did the retrieved chunks cover each fact the answer needs? | The most important retrieval number. A missed part can't be recovered by the generator |
| Precision@5 | How much of the context is relevant? | Low by design when a part has one gold chunk (at most 0.2 at k = 5). Compare across configs, not against 1.0 |
| F1@5 | Balance of the two | |
| MRR | Is the first gold chunk near the top? | Shows the reranker's effect, which P/R at a fixed k can miss |
| Hit@5 | % of questions where every part is covered | Strict; multi-fact questions are the hardest to pass |

**Two sources of these numbers:**

- `evaluate_rag.py` (Stage 1) retrieves with each mode directly.
- `score_retrieval.py` scores what each generated run actually retrieved (`retrieved_seen`). For the agent, that is the **union of unique chunks across all searches**, scored without a cutoff. The agent's set is usually larger, so `score_retrieval.py` reports chunks seen and chunks used (mean, p90) beside the metrics. Compare A1 with the matched-context baseline N2, not only with k = 5.

### 3.2 Faithfulness and context precision (judge-based)

`score_grounding.py` makes two judge calls per answer. Call these scores "RAGAS-style" in the report, not RAGAS. The contexts are the chunks **exactly as they appeared in the answer prompt** (`context_used`: the 2,000-character excerpts).

| Metric | What it checks | How it is computed |
| --- | --- | --- |
| Faithfulness | Every claim in the answer is supported by the context | Call 1: the judge lists the answer's factual claims and marks each supported or not, **without seeing the reference answer**. Score = supported / claims |
| Context precision | The chunks are relevant, and relevant ones rank high | Call 2: the judge marks each numbered chunk relevant or not, using the reference. `ctx_precision` = share relevant; `ctx_ap` = average precision (the quantity RAGAS's `LLMContextPrecisionWithReference` computes) |

- Answer relevance and context recall are not computed: part recall against gold already measures recall, and the rubric measures rightness.
- Only answerable questions with a real answer are scored. Raw judge output is cached with every score.
- `score_ragas.py`, in a separate environment, is an optional cross-check on about 10 answers.

### 3.3 Custom correctness rubric (LLM-as-judge)

`score_judge.py`, with the prompt `prompts/judge_correctness_v1.txt`:

| Score | Label | Rule |
| --- | --- | --- |
| 2 | Correct | All key facts present and correct, no contradicting claims |
| 1 | Partial | Main point right, but a key fact missing or a condition left out |
| 0 | Incorrect | Wrong main answer, any wrong number from `key_facts`, or a refusal on an answerable question |

Plus **numeric\_ok**: every number, rate, threshold and date in `key_facts` is correct. A wrong number forces a 0.

- **Out-of-scope questions** get a refusal check instead, scored correct when the system declines:
  - Status `refused` or `no_answer`, or an answer starting with `INSUFFICIENT_CONTEXT`, counts as declined without the judge.
  - `clarify` and run errors count as not declined.
  - Any other answer goes to the judge's short refusal prompt (`judge_refusal_v1.txt`).
- **Failed runs and non-answers count as wrong.** They score 0 without a judge call and stay in the denominator: a run error, an empty answer, or status `clarify`, `refused` or `no_answer` on an answerable question.
- **Report:** accuracy, mean correctness (0–2), numeric accuracy (on `numeric` = true), false refusals and clarifications on answerable questions, and the out-of-scope outcomes.

### 3.4 Efficiency

| Metric | How it is measured |
| --- | --- |
| Latency | Wall-clock seconds per question. For the agent, `SupervisorResult.timings` splits intake, route, plan, retrieval, audit, decide and answer; the naive record splits retrieval and generation. Report median and p90 |
| Tokens | Input and output tokens per question, summed over every LLM call (`tracking.py` patches `ollama.chat` and `ollama.Client.chat`; API adapters add their usage fields) |
| Cost | Tokens × the provider's listed price (note the date of the price list). Local Qwen: tokens and latency only, plus the hardware |

Compare latency within one machine. Every current run is on the cluster GPU (GTX 1080 Ti, node `bat`), so compare naive with agentic per generator.

## 4. LLM-as-judge design

One judge model, fixed across every run, gives every LLM-graded score: correctness, refusal checks, faithfulness and context precision.

### 4.1 Choosing the judge

- **Chosen judge:** `gemma4:12b` via Ollama (`configs/judge.yaml`, `provider: ollama`), run on the cluster GPU by `IS469_score.sh` after the generation job, so it has the card to itself.
- **Fixed across all runs.** If the judge changes, score differences mix judge behaviour with system behaviour. Re-judge everything if it ever changes: delete the `_judge` and `_grounding` score files, never the outputs.
- **Not one of the compared generators.** The generators are Qwen 3.5 4B, Claude Sonnet and gpt-oss-120b, which avoids self-preference bias. Gemma is a third family.
- **The limitation to state:** a 12B judge is weaker than the API generators it grades in Stage 3, and small judges can miss subtle numeric errors. The 30-pair validation in 4.3 decides whether its scores can be trusted.
- Record the model id in the report. Every score row stores `judge: "ollama:gemma4:12b"`.

### 4.2 Correctness judge prompt

The prompt is in `prompts/judge_correctness_v1.txt`. It gives the judge the question, the reference answer, the key facts and the candidate answer, plus the rubric in 3.3. It also has these rules:

- A wrong number always gives 0.
- Extra correct information is fine.
- Extra incorrect information that contradicts the reference gives 0.
- Judge meaning, not wording.

It asks for `{"reasoning", "numeric_ok", "score"}`, with the reasoning first, so the model reasons before it commits. The reply must match a JSON schema. Provider errors are retried with a pause. An unparseable reply leaves the row unscored, which counts as wrong; the raw output is kept on every row.

**The judge sees only text.** It has no access to the vector store; it grades the candidate against the reference answer and key facts. That is why every reference must be verified against the source chunks.

### 4.3 Validating the judge (to do)

1. Sample 30 (question, answer) pairs from the existing runs. Mix question types and systems, and include a few calculation and out-of-scope items.
2. Two team members score each pair with the rubric, blind to the judge's score, then agree on one human label.
3. Compute exact agreement on 0/1/2 and quadratic-weighted Cohen's kappa (`sklearn.metrics.cohen_kappa_score(human, judge, weights="quadratic")`), with a bootstrap 95% interval.
4. Target ≥ 80% exact agreement. Below that, read the disagreements, tighten the rubric wording, re-validate on 10 fresh pairs, and re-judge.
5. Report the agreement in the evaluation section.

Also spot-check faithfulness: read the 5 lowest- and 5 highest-scored answers.

## 5. Experimental design

Run three stages in order, then score. `run_all.py` runs the whole sequence (`--stages 1 2 3 4`, `--skip-api`, `--dry-run`). On the cluster, `IS469_eval.sh` runs Stages 1–2 and `IS469_score.sh` runs scoring. Every step resumes, so after a 2-hour time limit you resubmit the same job.

**Held constant in every stage:** the frozen dataset, chunking, the embedder, the ChromaDB index (recorded per run as the `chroma_db/` commit), the shared answer step, temperature 0 with thinking off, the fixed date 2026-10-28, and the judge.

### Shared settings

Every run reads its settings from one YAML in `rag_eval_approach/configs/`. `run.py` stores the config, its hash, the git commit, the corpus snapshot, the local model names and the machine in `<run_id>.meta.json`.

| Setting | Value |
| --- | --- |
| Retrieval depth k | 5 per query, for every system (N2 sets its own k, below) |
| Candidate pool | 10 (`CANDIDATES`) |
| RRF constant | 60 |
| Answer step | `ANSWER_SYSTEM`, temperature 0, `think=False`, `num_ctx` 8192, 2,000-character excerpt per chunk. Shared by N1, N2 and A1 (Section 0.4) |
| Agent limits (part of what "agentic" means here) | `max_steps` 4, `max_queries` 2, `max_chunks` 8, `audit_min_score` 0.5 (`a1_agentic_qwen.yaml`) |
| Tools | None (`tools=None`): no calculator or web search |
| Local models | All `qwen3.5:4b` (`src/config.py`). `generator.model: default` follows `SUPERVISOR_MODEL` |

### Stage 1: Retrieval ablation (RQ1, RQ2)

**Variable:** the retriever configuration. **Fixed:** the naive pipeline and the Qwen 3.5 4B generator.

| Config | Mode | Dense | BM25 | Reranker | Built from |
| --- | --- | --- | --- | --- | --- |
| R0 | `dense` | ✓ | | | `_vector_ranking` top k (naive baseline) |
| R1 | `bm25` | | ✓ | | `_bm25_ranking` top k (keyword reference point) |
| R2 | `hybrid` | ✓ | ✓ | | `_candidates` RRF order, top k |
| R3 | `dense_rerank` | ✓ | | ✓ | `_vector_ranking` top 10, then `_rerank` |
| R4 | `hybrid_rerank` | ✓ | ✓ | ✓ | `Retriever.search()` unchanged |

This is a 2×2 design (R0, R2, R3, R4) plus a BM25-only reference. The non-reranked modes have no 0–1 score, so `score` is the cosine, BM25 or RRF score. An agent on such a mode needs `audit_min_score` 0, and `run_all.py` sets that automatically.

**Procedure:**

1. `evaluate_rag.py --mode all` retrieves depth 10 for every answerable question, with no LLM, and scores the top 5. It saves `results/retrieval/<mode>.json`, with a summary for the primary set (`all`, `simple`, `multi`), `computation` and `all_with_computation`.
2. **Selection rule, written down before any run** (`run_all.select_mode`): carry R4 forward unless its part recall@5 is below R0's, or its median latency is over 3× R0's with no recall gain. Otherwise carry the mode with the highest part recall, ties broken by MRR. Only the primary set decides. The choice goes to `results/selected_mode.json`.
3. Generate answers with R0 (`s1_naive_r0_dense`) and with the selected mode (`n1_naive`), and score both.
4. The selection uses the same questions as the final report, so treat the ablation as descriptive. If R4 wins on retrieval but not on correctness, or R3 beats R4, report that.

**Table 1a: retrieval quality** (`analyze.py` → `table1a_retrieval_ablation`; primary set, plus a calculation row per mode)

| Config | Part recall@5 | P@5 | F1@5 | MRR | Hit@5 | Retrieval latency (ms, median) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 Dense … R4 Hybrid + rerank | | | | | | |

**Table 1b: end-to-end answer quality** (naive pipeline, Qwen 3.5 4B; rows of `runs_summary`)

| Config | Accuracy (95% CI) | Mean correctness | Numeric acc. | Faithfulness | Ctx precision |
| --- | --- | --- | --- | --- | --- |
| R0 | | | | | |
| Selected config (N1) | | | | | |

### Stage 2: Naive vs agentic RAG (RQ3)

**Variable:** the system. **Fixed:** the Stage 1 retriever, the shared settings and Qwen 3.5 4B.

| System | Config | Description | Why it is there |
| --- | --- | --- | --- |
| N1 Naive | `n1_naive` | One retrieval, top 5, one shared answer call | The fixed-pipeline baseline |
| N2 Naive, matched context | `n2_naive_matched` | N1 with k = ceil(A1's mean chunks in the answer prompt). `run_all.py` computes it; it came out at 6 (mean 5.26) | Separates "more context" from "agentic behaviour" |
| A1 Agentic | `a1_agentic_qwen` | The full stack of Section 0.3 | The system under test |

**What N2 does not control.** Beyond seeing more chunks, A1 also:

- drops chunks the auditor rejects;
- adds the intake's user facts and assumptions to the answer prompt;
- can stop early, give up or ask.

N2 controls only the number of chunks. State this in the report. If time allows, an **A1 without the auditor** run (the auditor approves everything) measures the auditor's own contribution.

**Running A1:**

- A fresh user per question, a fixed `today`, a temporary `UserStore` and no tools, all handled in `systems.py`.
- `run_all.py --repeats 1` is the default. The earlier three A1 repeats were identical at temperature 0, so they are not independent samples. If you rerun repeats, report mean ± range.
- All systems answer every record, out-of-scope ones included.

**Tables** (`analyze.py`):

| Table | Content | How |
| --- | --- | --- |
| 2a | Overall per system: accuracy with bootstrap CI, numeric accuracy, false refusals and clarifications, faithfulness, context precision, part recall, chunks seen/used, latency median/p90, tokens | `runs_summary` |
| 2b | Accuracy by question type, with calculation questions as their own row | `--types n1_naive n2_naive_matched a1_agentic_qwen` |
| 2c | Paired outcomes A1 vs N2 and A1 vs N1: win/loss table, exact McNemar, paired bootstrap CI of the difference, latency and token ratios. Printed for the primary set, the calculation questions and all answerable questions | `--pair a1_agentic_qwen n2_naive_matched --pair a1_agentic_qwen n1_naive` |
| 2d | Primary accuracy split by whether N1's retrieval covered every fact (Hit@5). Agentic RAG should help mainly where the first retrieval misses | `--stratify n1_naive` |
| 2e | Out-of-scope outcomes per run and `oos_group`, with Wilson CIs (Section 2.3) | Always written |

The off-diagonal cells of 2c are the questions that matter, and they feed the qualitative analysis.

### Stage 3: Generator LLM comparison (RQ4)

**Variable:** the LLM that writes the answer (naive) or that routes, decides and answers (agentic). **Fixed:** the Stage 1 retriever, the shared settings, the prompts and the judge.

**What a swap really changes.** Only the supervisor's `llm` callable is replaced. The intake gate, planner and auditor call local Ollama directly (`INTAKE_MODEL`, `PLANNER_MODEL`, `AUDITOR_MODEL`). So in an "agentic with Claude" run, Qwen 3.5 4B still decides refuse/clarify, writes planned queries and audits. Report it as "supervisor LLM swapped". Every provider sits behind one `llm(messages, schema)` adapter (`llm.py`; Anthropic, OpenAI-compatible including Groq, and Google).

| Run | Generator | System | Config |
| --- | --- | --- | --- |
| G1 / G2 | Qwen 3.5 4B | Naive / agentic | Reuse N1 and A1 |
| G3 / G4 | `claude-sonnet-5-5` (effort medium; no temperature) | Naive / agentic | `g_naive_anthropic`, `g_agentic_anthropic` |
| G5 / G6 | `openai/gpt-oss-120b` on Groq (temperature 0) | Naive / agentic | See the note below |

**Fix before running Stage 3:**

- `run_all.py` runs the `g_*_anthropic` and `g_*_openai` configs. `g_agentic_openai.yaml` (gpt-oss-120b on Groq) and `g_agentic_anthropic.yaml` still have `max_chunks: 6`, while A1 uses 8. So the agentic Stage 3 runs would differ from A1 in more than the model.
- The newer `g_*_gptoss120b` configs match A1 (`max_chunks` 8), but `run_all.py` does not run them. It also pairs only `g_*_anthropic` and `g_*_openai` in the analysis.
- **Pick one set** (set `max_chunks: 8` in the anthropic and openai configs, or switch `run_all.py` to the `gptoss120b` names) before the first Stage 3 run.
- The API runs need internet, so run them on a machine with outbound access, not on a compute node.

**Table 3: generator comparison**

| Generator | System | Accuracy (95% CI) | Numeric acc. | Faithfulness | OOS decline rate | Latency median | Cost per 100 questions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 3.5 4B | Naive / agentic | | | | | | local |
| Claude Sonnet | Naive / agentic (supervisor only) | | | | | | |
| gpt-oss-120b | Naive / agentic (supervisor only) | | | | | | |

Key reading for RQ4: compare the agentic gain (agentic minus naive accuracy) for each model. If the gain is largest for Qwen, agentic RAG compensates for a small model. If it is largest for the API models, the small model can't drive the agent well. Because the intake and auditor stay on Qwen in every agentic run, a small gain for the API models may come from the shared Qwen components.

### Run budget

| Stage | Generation runs | Notes |
| --- | --- | --- |
| 1 | 2 (R0, N1) | Plus 5 retrieval-only modes, with no LLM |
| 2 | 2 (A1, N2); N1 reused | A1 is about 3–4× N1's latency per question |
| 3 | 4 | API calls: 1 per record for the naive runs; for the agentic runs, a route call, 0–3 decide calls and an answer call per record |
| Scoring | | Per run: up to 80 judge rows (60 correctness + the out-of-scope refusal checks that need the judge), plus 2 grounding calls per answered question. Local judge: time, not money |

Cache every generation. Scoring never regenerates, and a judge change only deletes score files.

## 6. Qualitative analysis protocol

Read 10 cases in full, classify every failure in the test set with one taxonomy, and present 3 worked examples in the report.

### 6.1 Selecting the 10 cases

| Slot | Count | How to find them |
| --- | --- | --- |
| Agentic correct, naive wrong | 3 | Off-diagonal cell of Table 2c; prefer different question types |
| Naive correct, agentic wrong | 2 | The other off-diagonal cell; shows where the agent hurts |
| Both wrong | 2 | Correctness 0 for both; reveals corpus or retrieval limits |
| Retrieval perfect (Hit@5 = 1) but answer wrong | 1 | Isolates generation failures |
| Out-of-scope | 1 | One correct or failed decline, preferably a near-miss |
| Qwen wrong, API model right (Stage 3) | 1 | Shows what model size buys |

If a slot is empty, say so; that is itself a finding. Calculation questions may fill a slot; mark them as such.

### 6.2 Failure taxonomy

Tag every answer scored below 2 with **one primary code**, the earliest failure in the pipeline. One member tags and a second checks 10.

| Code | Failure | How to tell |
| --- | --- | --- |
| F1 | Retrieval miss | No gold chunk in the retrieved set |
| F2 | Ranked too low / filtered | Gold chunk retrieved but not in the answer prompt (outside the top k, below the reranker cutoff, or rejected by the auditor; note which) |
| F3 | Partial context | Some parts covered, but not all |
| F4 | Unsupported claim | Answer states something not in the context (low faithfulness) |
| F5 | Wrong number | numeric\_ok = false while the correct number was in the context (for calculation questions: an arithmetic error) |
| F6 | Misread context | Right chunk, wrong conclusion (missed a condition or exception) |
| F7 | Incorrect refusal | Declined an answerable question (`refused` or `no_answer`) |
| F8 | Failed refusal | Answered an out-of-scope question |
| F9 | Incomplete answer | Correct but missing a required part |
| F10 | Run failure | Error, timeout, empty answer, or `clarify` on an answerable question |

- F1–F3 are retrieval failures; F4–F6 and F9 are generation failures; F7–F8 are scope failures; F10 is a run failure.
- For agentic runs, the per-search log (`search_log`: audited and approved ids) shows whether the auditor rejected a gold chunk.

**Table 4: failure counts.** Rows F1–F10; columns Naive (Qwen), Agentic (Qwen) and Agentic (best API model).

### 6.3 Worked-example template

```markdown
### Case <qid>: <short title, e.g. "Agentic recovers a missed REIT ETF passage">
**Question:** ...
**Type / source:** multi-part / reit_etf_guide
**Reference answer (short):** ...

| | Naive | Agentic |
|---|---|---|
| Retrieved chunks (ids, gold marked ✓) | ... | ... |
| Answer (trimmed) | ... | ... |
| Correctness / numeric_ok | ... | ... |
| Faithfulness | ... | ... |
| Failure code | ... | ... |

**Diagnosis:** 2–3 sentences on what went right or wrong, and at which step (retrieval vs generation).
**What it shows:** one sentence linking the case to a research question.
```

## 7. Statistical caution for a small dataset

With 50 primary questions, one question moves accuracy by 2 points. Report uncertainty on every headline number, and compare systems per question, not only by averages.

1. **Report n everywhere.** For any category with n < 5, give the raw count ("3/4"), not a percentage.
2. **Bootstrap 95% confidence intervals** for accuracy: resample questions with replacement 1,000 times (`analyze.bootstrap_ci`, seed 0).
3. **Paired comparison for A1 vs N2 and A1 vs N1.**
   - Both systems answer the same questions, so count wins, losses and ties (Table 2c).
   - The exact McNemar test (a binomial test on the discordant pairs) is optional: at n = 50 it reaches p < 0.05 only when the off-diagonal is very lopsided, roughly 6–0 or 10–2.
4. **Paired bootstrap for the difference.** Resample question indices once per iteration and compute A1 minus naive on the same sample. If the 95% interval includes 0, write "no clear difference".
5. **Repeats.** At temperature 0 the A1 repeats were identical, so one run is the default. Say so as a limitation. Re-judge about 10 answers once to confirm the judge is stable.
6. **Wording rules for the report** (`analyze.wording`):

| Observed accuracy gap | How to describe it |
| --- | --- |
| ≤ 2 questions | "comparable"; no claim of a winner |
| 3 or more questions, CI includes 0 | "slightly different, but within noise" |
| 3 or more questions, CI excludes 0 | "higher" or "lower", with the interval |

7. **Don't over-read per-category results.** A category with 5 questions swings 20 points on one answer. Use the per-category tables to describe patterns and pick cases, not to claim significance. The 10 calculation questions are descriptive for the same reason.
8. **Out-of-scope rates** are small proportions (n = 20, often near 0% or 100%), so they use Wilson intervals rather than the bootstrap, which collapses to a single point at 20/20.

## 8. Trade-off discussion framework

Judge each alternative against its baseline on four dimensions, then state whether its quality gain justifies its extra cost.

### 8.1 Dimensions

| Dimension | Measured by | Direction |
| --- | --- | --- |
| Quality | Accuracy, numeric accuracy, faithfulness, out-of-scope decline rate | Higher is better |
| Latency | Median and p90 seconds per question | Lower is better; p90 matters for a chatbot user |
| Cost | Tokens per question; USD per 100 questions for API models | Lower is better |
| Complexity | Extra components, prompt engineering, failure modes added (qualitative) | Lower is better |

### 8.2 Summary table for the report (Table 5)

| Alternative vs baseline | Δ accuracy (pp, 95% CI) | Δ faithfulness | Latency ratio (×) | Token ratio (×) | Complexity added | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid vs dense | | | | | BM25 index | |
| Rerank vs no rerank | | | | | Cross-encoder model | |
| Agentic vs matched-context naive (Qwen) | | | | | Intake gate, planner, auditor, supervisor loop | |
| API model vs Qwen (agentic) | | | | | API dependency, cost | |

### 8.3 Decision rule (agreed before seeing results)

- **Worth it:** an accuracy gain over the matched-context baseline N2 whose 95% CI excludes 0, with a latency ratio ≤ 3×.
- **Situational:** the gain is mainly on one question type (e.g. multi-step), or the latency ratio is > 3×. Recommend routing: the cheaper system by default, the costlier one for that type.
- **Not worth it:** no clear accuracy gain (the CI includes 0), or faithfulness drops.

Also report **extra correct answers per extra second** = Δ correct answers ÷ Δ total latency over the test set.

## 9. Implementation

Generation and scoring are separate steps, so a judge fix never forces a regeneration.

### 9.1 Design decisions taken from the codebase check (28 Oct)

| # | Finding | What was done |
| --- | --- | --- |
| 1 | The old naive generator (`src/inference/inference.py`, `RAGGenerator`) used its own prompt, temperature 0.1 and whole chunks | One shared answer step (`answer.py`) reuses the supervisor's prompt, limits and excerpting. `RAGGenerator` is not used for results |
| 2 | "Agentic" is the whole supervisor stack, which can clarify or refuse | A1 is described that way (Section 0.3). Statuses are scored (Section 3.3). Fresh user, fixed date and a temporary store per question |
| 3 | `Retriever.search()` always fuses and reranks | `ModeRetriever` builds R0–R3 from the stage methods. Production `search()` is unchanged |
| 4 | Only the supervisor's `llm` is swappable | The Stage 3 limitation is stated; one adapter per provider |
| 5 | `SupervisorResult` exposes no chunk ids or token counts | `systems.py` wraps retrieval, the auditor and the answer prompt; `tracking.py` patches both Ollama chat entry points |
| 6 | `chroma_db/` is git-tracked, and `UserStore` writes into it by default | Temporary `UserStore` path. Every run records the `chroma_db/` commit and whether it was dirty. `git restore -- chroma_db` after scripts that open the collection |
| 7 | The cluster has a 2-hour job limit | `run.py` and every scorer resume. `--start/--end` slice the dataset |

### 9.2 Libraries

| Purpose | Library |
| --- | --- |
| Vector store, BM25, reranker, embedder | `chromadb`, `rank_bm25`, `sentence-transformers` |
| Local LLMs and judge | `ollama` |
| Config files | `PyYAML` |
| API generators | `anthropic`, `openai` (also for Groq), `google-genai` (`requirements-api.txt`) |
| Analysis | `pandas`, `numpy`, `scipy` (binomial test for McNemar) |
| Optional RAGAS cross-check | `ragas` in a separate environment (`requirements-scoring.txt`) |

### 9.3 Repo structure and logging

```text
src/evaluation/
  eval_questions.json         # the dataset (Section 2)
  qid_map.json                # old Q-ids -> S/M/A/C/O ids
  agent_eval/                 # the agentic team's suites (robustness incl. P01-P05, safety, trajectory, task success)
  rag_eval/
    RAG_EVAL_PLAN.md          # this file
    run_all.py                # Stages 1-4 in order, with the Stage 1 selection rule; resumable
    analyze.py                # report tables: bootstrap CIs, paired tests, by type, strata, out-of-scope (no model calls)
    db_check/
      facts.py                # loads the chunks; snippet matching (norm, matching_ids)
      check_dataset.py        # validates the dataset against the corpus (writes nothing)
      label_gold.py           # fills gold_passages from the verbatim facts (--write)
      apply_qid_map.py        # renames old qids in an older results folder
    rag_eval_retrieval/
      retrieval_modes.py      # ModeRetriever: R0-R4
      metrics.py              # part recall / precision / F1 / MRR / hit
      evaluate_rag.py         # Stage 1 retrieval-only scores -> results/retrieval/<mode>.json
    rag_eval_approach/
      answer.py               # the ONE shared answer step
      systems.py              # NaiveSystem, AgenticSystem: .answer(question, qid, today) -> record
      llm.py, llm_anthropic.py, llm_openai.py, llm_google.py   # one llm(messages, schema) shape per provider
      tracking.py             # token counts over every Ollama / API call
      run.py                  # config + dataset -> results/outputs/<run_id>.jsonl (+ .meta.json); resumable
      results_io.py           # paths, loaders, analysis groups (is_core, SECONDARY_QTYPES), oos_outcome
      score_retrieval.py      # retrieval metrics on what each run retrieved (no LLM)
      score_judge.py          # correctness + refusal checks -> results/scores/<run_id>_judge.jsonl (cached)
      score_grounding.py      # faithfulness + context precision -> <run_id>_grounding.jsonl (cached)
      score_ragas.py          # optional RAGAS cross-check (separate environment)
      check_judge.py          # grades five canned cases before any real scoring
      env.py                  # loads API keys from .env (never committed)
      configs/                # s1_naive_r0_dense, n1_naive, n2_naive_matched, a1_agentic_qwen, g_*, judge.yaml
      prompts/                # judge_correctness_v1, judge_refusal_v1, grounding_faithfulness_v1, grounding_context_precision_v1
    results/                  # retrieval/, outputs/ (never edited by hand), scores/, report_assets/, selected_mode.json
IS469_eval.sh                 # cluster job: Stages 1-2 (run_all.py --stages 1 2 --skip-api by default)
IS469_score.sh                # cluster job: scoring with the local Ollama judge
```

**Each line of `outputs/<run_id>.jsonl`** holds one question's record:

- `run_id`, `qid`, `config_hash`, `git_commit`, `timestamp`, `generator`, `system` and `status`;
- `answer` (the agent's draft, without the Assumptions/Sources footer);
- `retrieved_seen` (every chunk returned, in order of first appearance) and `context_used` (the chunks in the answer prompt, as excerpted);
- `searches`, `latency_s` per stage and `tokens`, plus `error` and `wall_s`;
- agent records also have `stop_reason`, `queries`, `search_log` and `trace`.

**Rules:**

- A run id belongs to one configuration. `run.py` refuses to continue if the config hash changed; give the changed config a new run id.
- The hash does **not** cover question text or code. When a question is reworded, delete its rows from `results/outputs/*.jsonl` and `results/scores/*.jsonl`, so `run.py` answers it again. When a reference answer changes, delete only its `_judge` and `_grounding` rows.

  ```bash
  python -c "
  import glob, json
  for f in glob.glob('src/evaluation/rag_eval/results/outputs/*.jsonl') + glob.glob('src/evaluation/rag_eval/results/scores/*.jsonl'):
      drop = {'S10'} | ({'M02', 'M12'} if f.endswith(('_judge.jsonl', '_grounding.jsonl')) else set())
      L = open(f, encoding='utf-8').read().splitlines(True)
      K = [l for l in L if not l.strip() or json.loads(l).get('qid') not in drop]
      if len(K) != len(L): open(f, 'w', encoding='utf-8').write(''.join(K)); print(f, len(L) - len(K))
  "
  ```

  Running this is the pending step before the next Stage 1–2 run.
- Errors are logged per question with status `error`, and the run carries on. Report error counts.

### 9.4 Team roles

| Member | Owns | Also does |
| --- | --- | --- |
| A | Dataset: `needs_review` checks, gold labelling and audit | Second-checks failure codes |
| B | Harness: `run.py`, `answer.py`, `systems.py`, `retrieval_modes.py`, logging; Stage 1 | Retrieval metrics |
| C | Judge and grounding scorer, judge validation, API adapters | Stage 3 runs |
| D | Stage 2 runs, `analyze.py`, statistics, qualitative analysis lead | Report tables and figures |

Everyone reviews about 4 flagged questions and scores about 15 judge-validation pairs (two scorers per pair).

### 9.5 Remaining timeline (suggested, to 10 Nov)

| Dates | Milestone | Done when |
| --- | --- | --- |
| 5–6 Nov | Drop the stale rows (9.3); cluster: Stage 1 rerun, Stage 2 resume (S10, C01–C10); Stage 4 scoring; check the `needs_review` records | Table 1a/1b/2a–2e filled on 80 records |
| 6–7 Nov | Judge validation on 30 pairs; fix the Stage 3 configs and run them | Agreement reported; Table 3 filled |
| 7–8 Nov | Failure coding (F1–F10), 10 qualitative cases, trade-off table | Tables 4 and 5, 3 worked examples |
| 8–9 Nov | Write the evaluation section; final checks | Draft reviewed by 2 members |
| 10 Nov | Submit (G2 on 13 Nov gives 3 days of buffer) | |

No new configs after 7 Nov. Anything later goes into the report as a limitation.

## 10. Deliverables and evaluation section outline

### 10.1 Tables and figures

| ID | Content | From |
| --- | --- | --- |
| Table 0 | Dataset summary: groups, types, sources, difficulty; judge–human agreement | Sections 2, 4.3 |
| Table 1a / 1b | Retrieval ablation (R0–R4); end-to-end quality (R0 vs selected) | Stage 1 |
| Table 2a–2e | N1 vs N2 vs A1: overall, by type, paired, stratified, out-of-scope | Stage 2 |
| Table 3 | Generator comparison | Stage 3 |
| Table 4 | Failure code counts per system | Section 6.2 |
| Table 5 | Trade-off summary with verdicts | Section 8 |
| Figure 1 | Part recall@5 and MRR per retriever config | Table 1a |
| Figure 2 | Accuracy by question type, N1 vs N2 vs A1 | Table 2b |
| Figure 3 | Quality vs latency: one point per system × generator | Tables 2a, 3 |

### 10.2 Evaluation section outline

1. **Setup** (½ page): the system (Section 0, condensed), dataset and groups (Table 0), gold labelling, shared settings, the intake-safe wording note, the judge and its validated agreement.
2. **Metrics** (½ page): one line per metric and why it fits a tax chatbot; why BLEU/ROUGE were not used.
3. **Stage 1** (½–1 page): Tables 1a and 1b, Figure 1. Which component helped retrieval, and whether it carried through to answers.
4. **Stage 2** (1 page): Tables 2a–2e, Figure 2. Lead with the paired result and its CI; then the calculation and out-of-scope groups.
5. **Stage 3** (½–1 page): Table 3, the agentic gain per model, and the supervisor-only caveat.
6. **Qualitative analysis** (1 page): Table 4 and 3 worked examples.
7. **Trade-offs** (½ page): Table 5 and Figure 3, applying the rule in Section 8.3.

Open each subsection with its finding and number (e.g. "Agentic RAG answered 5 more primary questions correctly than matched-context naive RAG (+10 pp, 95% CI …)"), then the table.

## Limitations and scope

- **Single-turn only.** The agent supports clarification turns and user profiles. The questions are worded so that the intake gate never clarifies (Section 2.3), so clarification quality is not measured here; it belongs to the agentic team's suites.
- **No tools.** The calculator is off, so the calculation questions measure the answer LLM's own arithmetic.
- **Supervisor-only swap in Stage 3.** The intake, planner and auditor stay on Qwen 3.5 4B.
- **Time-sensitive answers.** Results depend on the corpus snapshot (scraped 14 Oct 2026; e.g. the YA 2026 rebate) and on the fixed date 2026-10-28.
- **Same questions for selection and reporting.** Stage 1's selection and the final numbers use one test set, so the ablation is descriptive.
- **Judge.** A 12B local judge with validation pending (Section 4.3).
- **Not covered:** citation accuracy (whether `[Doc n]` points at the right chunk), multi-turn conversations, and a fully API-driven agent.
