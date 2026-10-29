# IS469 RAG Evaluation Plan (revised for the current codebase)

Oct 28, 2026 · @Joel

## Summary and key decisions

Evaluate the RAG part in three stages: a retrieval ablation, then naive vs agentic RAG, then a generator LLM comparison. Each stage carries its winner forward, so the team runs about 10 generation runs instead of a full grid. Stage 2 includes a naive baseline given the same amount of context as the agent, so agentic RAG isn't credited merely for seeing more chunks. One primary metric, accuracy (correctness = 2), carries the headline claims; all other metrics are diagnostic.

**Should we add retrieval methods one by one? Yes.** Do it as Stage 1, as a 2×2 design: dense vs hybrid, each with and without the cross-encoder reranker. Dense-only with no reranker is the naive baseline. Retrieval metrics need no LLM, so all configurations are cheap to score. This answers the slides' point that each technique must earn its place.

**Is the dataset big enough? Just enough, with caveats.** Dataset v2 (`src/evaluation/eval_questions.json`) has 35 scored questions: 16 single-fact, 15 multi-part and 4 out-of-scope, plus 5 retrieval-only paraphrases (40 records). One question is worth 2.9 percentage points, so gaps under about 12 points (4 questions) are likely noise, and only gaps of about 25 points or more can be shown reliably. Use all 35 as the test set. The 5 paraphrases give retrieval something to discriminate (Section 2.3).

**Gold passages: auto-labelled, then audited.** `label_gold.py` marks the chunks that contain each verbatim fact and stores their ids. The corpus will not be re-ingested, so the ids stay valid. You need them for retrieval Precision/Recall/F1 and for the Stage 2 stratification. Verify the 11 `needs_review` reference answers first, then hand-check about 15 questions' labels (Section 2.4).

**Three fixes from checking the plan against the code.** These are explained in Section 9.1:
1. The existing naive generator (`src/inference/inference.py`) is not comparable to the agent's answer step. Both systems must share one answer function.
2. "Agentic RAG" in this repo is the whole supervisor stack (decomposer, auditor, supervisor loop, user profile), not retrieval as a single tool. The report must say so, and the harness must handle clarify, refuse and profile effects.
3. The retriever has no dense-only, BM25-only or no-rerank modes, which Stage 1 needs.

## 1. Objectives and research questions

The RAG evaluation answers four questions. Each maps to one stage and one table in the report.

| ID | Research question | Stage | Primary metric |
| --- | --- | --- | --- |
| RQ1 | Which retrieval configuration finds the right IRAS passages most reliably? | 1 | Recall@5, MRR |
| RQ2 | Do hybrid search and reranking improve final answers, not just retrieval scores? | 1 | Correctness, faithfulness |
| RQ3 | Does agentic RAG give better answers than naive RAG, including a naive baseline given the same amount of context, and is the gain worth the extra latency and tokens? | 2 | Correctness, latency |
| RQ4 | How much does the supervisor/generator LLM matter, and does agentic RAG help a small model (Qwen 3.5 4B) more than larger ones? | 3 | Correctness, faithfulness |

A secondary objective is to find *where* each system fails, by question type and source, through the qualitative analysis (Section 6).

## 2. Dataset preparation

Dataset v2 is `src/evaluation/eval_questions.json` (40 records: 35 scored + 5 paraphrases). It was built from the original 40 questions, with the unnatural ones reworded, duplicates removed, four new gap questions and four out-of-scope questions added, and tags, reference answers and key facts for every record. Its `facts` field keeps the original format (a list of parts, each a list of alternative verbatim strings). Commit it and freeze it before any comparison run.

### 2.1 Record format

Each record is one question. `facts` is a list of parts: each part is one fact the answer needs, written as one or more alternative verbatim strings from the corpus. `gold_passages` is filled by `label_gold.py` with one list of chunk ids per part (the ids below are illustrative; real ids look like `web:<10 hex characters>:<n>` or `<pdf name>:<n>`).

```json
{"qid": "Q22", "type": "multi", "tier": "webpage",
 "qtype": "multi-part", "source": "online_guide", "difficulty": "hard", "numeric": true,
 "answerable": true,
 "question": "What penalty applies if my company pays its tax after the due date, and what is the minimum monthly GIRO amount if it spreads payment under the instalment plan?",
 "reference_answer": "A 5% late payment penalty is imposed on the unpaid tax. Under a GIRO instalment plan, the minimum monthly deduction is $50.",
 "key_facts": ["5%", "$50"],
 "facts": [["A 5% late payment penalty will be imposed on the unpaid tax"],
           ["minimum monthly GIRO deduction amount of $50"]],
 "gold_passages": [["web:3fa9c1d27b:4"], ["web:3fa9c1d27b:9"]],
 "needs_review": false}
```

`key_facts` lists the numbers, dates or terms a correct answer must contain. The judge uses it for the numeric check.

### 2.2 Tags

Dataset v2 is already tagged on these fields. Have a second member check the tags independently and settle disagreements together (about 30 minutes).

| Field | Values | Rule |
| --- | --- | --- |
| qtype | definition, number, procedure, multi-part, edge-case, out-of-scope | `number` = a single fact hinging on a rate, threshold, date or amount. `multi-part` = needs facts from 2+ passages. `out-of-scope` = not answerable from the corpus |
| source | online\_guide, reit\_etf\_guide, insurer\_guide, multiple, none | Where the answer lives. The corpus is the scraped IRAS web pages (`data/raw/iras_corporate_tax.jsonl`) plus two PDFs in `data/manual/`: the REIT ETF guide and the general insurers' IBNR guide |
| difficulty | easy, hard | `hard` = multi-part, needs reasoning across sections, or has a tricky condition |
| numeric | true, false | `true` if the answer hinges on a number, rate, threshold or date, whatever the qtype. Drives numeric accuracy across all questions |

### 2.3 Is the dataset enough? Dataset v2 plus paraphrases

Dataset v2 has 35 scored questions (31 answerable + 4 out-of-scope) and 5 paraphrases. The set was sized as the smallest that still fills every table in this plan (see the task section at the end for the reasoning).

| Bucket | Target | Dataset v2 | Note |
| --- | --- | --- | --- |
| Each of the 3 sources | ≥ 3 (descriptive only, no significance claims) | online guide 25, REIT ETF guide 3, insurer guide 3 | Met |
| numeric (flag) | ≥ 10 | 19 | Met |
| multi-part | ≥ 10 | 15 | Met |
| single-fact types (number, definition, procedure, edge-case) | 3–5 each | 5, 4, 3, 4 | Met |
| out-of-scope | 4 | 4 | Four is too few for a precise refusal rate, so report refusals as counts (e.g. 3/4) |
| paraphrased / colloquial | 5 (retrieval only) | 5 | Met: 3 single-part, 2 multi-part |

**Why paraphrases:** the original questions were written from verbatim passages, so they share many words with the text. Dense and BM25 retrieval may both score near-perfectly (a ceiling effect), hiding any difference between configurations. The 5 paraphrases are everyday rewordings (e.g. "my company made no profit this year, still need to do the ECI thing?"), tagged `paraphrase_of: Qxx`. They reuse their original's `facts`, and are scored on retrieval metrics only, so they add no judge cost. Report them separately (`evaluate_rag.py --paraphrases`), and resample by original question in the statistics (Section 7). On a first run they scored lower than the originals (single-part hit@1 67% vs 88%), so they do separate the retrieval configurations.

### 2.4 Labelling gold passages

Label gold passages automatically from the verbatim `facts`. `label_gold.py` scans every chunk in ChromaDB and, for each part of each question, records the ids of the chunks whose text contains any of that part's alternative strings. It matches on heading path plus text, ignoring case, whitespace and apostrophe style, as `fact_ranks()` in `evaluate.py` does. Chunk ids (`web:<10 hex characters>:<n>` and `<pdf name>:<n>`) stay valid because the corpus will not be re-ingested; if it ever is, re-run the script. The script also flags facts that match no chunk, or more than 5.

Steps:

1. Run `python label_gold.py --db <path> --collection <name> --data src/evaluation/eval_questions.json`.
2. Fix every flagged fact by hand: replace fragments with fuller sentences from the corpus. Also verify the 11 records marked `needs_review` (reference answers and key facts) against the source chunks.
3. Audit: hand-check about 15 questions (at least 3 per source, mostly multi-part). Open each matching chunk and confirm it contains the information, not just a heading or a generic phrase like "continues to be dormant". If more than 3 of the 15 are wrong, tighten the fact strings.
4. Out-of-scope questions keep empty gold lists. Record the corpus snapshot (git commit of `chroma_db/`, scrape date) with the dataset.

Estimate: 2–3 hours split across the team for the flagged questions and the audit. Freeze the dataset before the first comparison run.

## 3. Metrics

Use four groups of metrics: retrieval (no LLM needed), RAGAS (LLM-graded), a custom correctness rubric, and efficiency. Skip BLEU/ROUGE, since the slides note they reward word overlap rather than correctness.

**Primary metric: accuracy**, the % of questions with correctness = 2 (Section 3.3). All headline claims about which system or configuration is better rest on this one number. Everything else is diagnostic: it explains *why* accuracy differs and is not used to declare a winner, which avoids cherry-picking among many scores.

### 3.1 Retrieval metrics (vs gold passages)

Compute on the top-k chunks passed to the generator (k = 5 unless you use another value), for answerable questions only. Gold is stored as **parts**: each part is one fact the answer needs, and a part is covered if the retrieved chunks include any chunk id listed for it. Part-level scoring avoids penalising a system that retrieves a duplicate or equivalent chunk.

```text
Recall@k    = parts covered / parts            (= cov@k in evaluate.py)
Hit@k       = 1 if every part is covered       (= all@k in evaluate.py)
Precision@k = gold chunks among the top k / k
F1@k        = 2 * P * R / (P + R)
MRR         = mean over questions of 1 / (rank of the first gold chunk)
```

| Metric | What it tells you | Note |
| --- | --- | --- |
| Part recall@5 | Did the retrieved chunks cover each fact the answer needs? | The most important retrieval number. A missed part can't be recovered by the generator |
| Precision@5 | How much of the context is relevant? | Low by design when a part has 1 gold chunk (max 0.2 at k = 5). Compare across configs, not against 1.0 |
| F1@5 | Balance of the two | Report as the slides ask |
| MRR | Is the first gold chunk near the top? | Shows the reranker's effect, which P/R at fixed k can miss. `evaluate.py` currently computes it for simple questions only; extend it to all |
| Hit@5 | % of questions where every part is covered | Strict; multi-part questions are the hardest to pass |

For agentic RAG, the retrieved set is the **union of unique chunks returned across all retrieval calls** (before the auditor filters them). This set is usually larger than naive RAG's fixed 5, so agentic recall is inflated by volume. Always report **chunks seen per question** (mean and p90) beside the retrieval metrics, and compare agentic RAG against the matched-context naive baseline N2 (Section 5, Stage 2), not only against the k = 5 baseline. Also log the chunks that actually went into the answer prompt (Section 9.3).

```python
def retrieval_metrics(retrieved_ids, gold_parts, k=5):
    """gold_parts: one list of chunk IDs per part (each part = one fact the answer needs)."""
    top = retrieved_ids[:k]
    covered = [bool(set(top) & set(part)) for part in gold_parts]
    gold_all = {c for part in gold_parts for c in part}
    r = sum(covered) / len(gold_parts)
    p = len(set(top) & gold_all) / k
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rr = next((1 / (i + 1) for i, c in enumerate(retrieved_ids) if c in gold_all), 0.0)
    return {"p": p, "r": r, "f1": f1, "rr": rr, "hit": int(all(covered))}
```

### 3.2 RAGAS metrics

Compute with the `ragas` library, using the fixed judge model (Section 4). Inputs per question: `user_input` (question), `retrieved_contexts` (the chunk texts **exactly as they appeared in the answer prompt**, including any truncation), `response` (answer), `reference` (reference answer). Scoring faithfulness against text the generator never saw would be wrong.

| Metric | What it checks | RAGAS class (pin your version; names change between releases) |
| --- | --- | --- |
| Faithfulness | Every claim in the answer is supported by the retrieved context | `Faithfulness` |
| Answer relevance | The answer addresses the question | `ResponseRelevancy` (needs an embedding model; reuse Qwen3-Embedding-0.6B) |
| Context precision | Retrieved chunks are relevant, and relevant ones rank high | `LLMContextPrecisionWithReference` |
| Context recall | Retrieved chunks cover everything in the reference answer | `LLMContextRecall` |

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

ds = EvaluationDataset.from_list(rows)  # rows: dicts with the 4 fields above
result = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
                  llm=judge_llm, embeddings=judge_emb)
result.to_pandas().to_csv("results/ragas_<run_id>.csv")
```

Run RAGAS only on answerable questions. Faithfulness is undefined for a refusal. Expect roughly 10 or more judge calls per question per run: context precision calls the judge once per retrieved chunk, so agentic runs with larger contexts cost more. Cache every score.

### 3.3 Custom correctness rubric (LLM-as-judge)

RAGAS checks grounding and relevance, but not whether the answer is *right*. A faithful answer built on the wrong chunk is still wrong. Add a correctness score against the reference answer:

| Score | Label | Rule |
| --- | --- | --- |
| 2 | Correct | All key facts present and correct, no contradicting claims |
| 1 | Partial | Main point right, but a key fact missing or a condition left out |
| 0 | Incorrect | Wrong main answer, any wrong number from `key_facts`, or a refusal on an answerable question |

Plus one binary flag, **numeric\_ok**: every number, rate, threshold and date in `key_facts` appears correctly. A wrong number forces a score of 0, because a wrong tax rate is a wrong answer.

For out-of-scope questions, score **refusal correctness** instead: 1 if the system declines or says the corpus does not cover it, 0 if it answers anyway. This can be detected without the judge: the supervisor returns status `no_answer` (or `refused`), and the naive answer starts with `INSUFFICIENT_CONTEXT`.

Report: **accuracy** (% scored 2) as the primary metric, mean correctness (0–2), numeric accuracy (% numeric\_ok on questions with `numeric` = true), and refusal accuracy.

**Failed runs and non-answers count as wrong.** Give correctness 0 and keep in the denominator when:
- a run errors (tool-call failure, step-limit hit, timeout) or returns an empty answer;
- the agent returns status `clarify` (it asks the user a question instead of answering) or `refused` on an answerable question;
- the agent returns `no_answer` on an answerable question (this is also an incorrect refusal, code F7 in Section 6.2).

Report the count of each per system. Dropping these would flatter the weaker generator, and Qwen 3.5 4B will likely fail more often as an agent.

### 3.4 Efficiency

| Metric | How to measure |
| --- | --- |
| Latency | Wall-clock seconds per question with `time.perf_counter()`. For the agent, `SupervisorResult.timings` already splits intake, retrieval, audit, decide and answer; log the naive system's retrieval and generation the same way. Report median and p90 |
| Tokens | Input and output tokens per question, summed over every LLM call (intake, audit, decide, answer). Ollama returns `prompt_eval_count` and `eval_count`; API models return usage fields (Section 9.3) |
| Cost | Tokens × the provider's listed price (note the date of the price list). For local Qwen, report tokens and latency, and note the hardware |

Latency for local Qwen and for API models isn't directly comparable (different hardware and network). Compare naive vs agentic within each generator, and note the hardware (laptop CPU, or the cluster GPU).

## 4. LLM-as-judge design

Use one strong judge model, at temperature 0, for every LLM-graded score (RAGAS and the correctness rubric) across every run.

### 4.1 Choosing the judge

- **Fixed across all runs.** If the judge changes, score differences mix judge behaviour with system behaviour, and the comparison breaks.
- **Not one of the compared generators.** LLM judges tend to rate their own outputs higher (self-preference bias). Since Stage 3 compares an Anthropic and an OpenAI model, the cleanest choice is a strong model from a third provider (e.g. a Gemini model). If you only have Anthropic and OpenAI access, use a model from one family that is *not* in your comparison, and name this as a limitation.
- **Strong enough.** Qwen 3.5 4B is too weak to judge faithfulness reliably. Don't use it as the judge.
- Record the exact model ID and date in the run config.

### 4.2 Sample correctness judge prompt

```text
You are grading answers from a Singapore corporate tax chatbot against a reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

KEY FACTS (each must appear correctly for full marks):
{key_facts}

CANDIDATE ANSWER:
{response}

Grade the candidate on correctness only. Ignore style, length and citations.

Rubric:
2 = Correct: all key facts present and correct; no claim contradicts the reference.
1 = Partial: main point correct, but a key fact or condition is missing.
0 = Incorrect: main answer wrong, ANY number/rate/threshold/date wrong, or the candidate refuses although the reference gives an answer.

Rules:
- A wrong number always gives 0, even if the rest is correct.
- Extra correct information is fine. Extra incorrect information that contradicts the reference gives 0.
- Judge meaning, not wording.

Return ONLY this JSON:
{"reasoning": "<one or two sentences>", "numeric_ok": true|false, "score": 0|1|2}
```

Put `reasoning` before `score` so the model reasons before it commits. Parse the JSON and retry once on a parse failure. Log the raw judge output.

For out-of-scope questions, use the status check in Section 3.3, and a short judge prompt only for naive answers that don't start with `INSUFFICIENT_CONTEXT` (returns `refused: true|false`).

### 4.3 Validating the judge

1. Sample 30 (question, answer) pairs from the first completed runs (Stage 1 naive answers plus early agentic answers). Mix question types and include a few out-of-scope refusals.
2. Two team members score each pair with the same rubric, without seeing the judge's score. Settle disagreements together to produce one human label.
3. Compare judge vs human: exact agreement on the 0/1/2 score and weighted Cohen's kappa (`sklearn.metrics.cohen_kappa_score(human, judge, weights="quadratic")`). With only 30 pairs, add a bootstrap 95% interval for kappa and report it.
4. Target ≥ 80% exact agreement. A kappa point estimate of 0.6 on 30 pairs still has a wide interval, so treat the interval as the finding. Below target, read the disagreements, tighten the rubric wording, and re-validate on a fresh 10 pairs.
5. Report the agreement numbers in the evaluation section. This is a cheap way to show the LLM scores can be trusted.

Do a lighter spot check for RAGAS faithfulness: read 10 answers with the lowest and highest faithfulness and confirm the scores make sense.

Some reference answers were drafted by an LLM. A person must verify every reference answer and key-facts list against the source documents before the judge runs, especially the 11 records flagged `needs_review`. Otherwise the judge grades against unverified answers.

## 5. Experimental design

Run three stages in order. Each stage carries its winner forward, giving about 10 generation runs in total instead of 30+.

**Held constant in every stage:** the frozen dataset, chunking (size and overlap), Qwen3-Embedding-0.6B, the ChromaDB index (record the git commit of `chroma_db/`), the answer function and prompt (below), generation temperature 0 with thinking off, a fixed date passed to the agent, and the judge model.

### Shared settings: one config for all systems

All systems read these values from one config file (`src/evaluation/rag_eval/configs/*.yaml`) and log them in every run.

| Setting | Value |
| --- | --- |
| Per-query retrieval depth k | 5 for all systems |
| Reranker candidate pool | 10 for all systems (the existing `CANDIDATES`) |
| RRF constant | 60 |
| Answer function | One shared function for N1, N2 and A1: the supervisor's `ANSWER_SYSTEM` prompt, temperature 0, `think=False`, each chunk cut to 1,200 characters (`DOC_CHARS`) |
| Agent-only limits (part of what "agentic" means here) | `max_steps` 4, `max_queries` 2, `max_chunks` 6, `audit_min_score` 0.5 |
| Tools | None registered (`tools=None`), so no calculator or web search is used |

### Stage 1: Retrieval ablation (RQ1, RQ2)

**Variable:** retriever configuration. **Fixed:** naive pipeline, Qwen 3.5 4B generator.

| Config | Dense | BM25 | Reranker | Role |
| --- | --- | --- | --- | --- |
| R0 | ✓ |  |  | Naive baseline |
| R1 |  | ✓ |  | Reference point: shows how much exact keyword matching alone gets you |
| R2 | ✓ | ✓ |  | Effect of hybrid only |
| R3 | ✓ |  | ✓ | Effect of reranking only |
| R4 | ✓ | ✓ | ✓ | Full stack (today's `Retriever.search()`) |

This is the 2×2 design (R0, R2, R3, R4) plus a BM25-only reference. Hybrid fusion: Reciprocal Rank Fusion with constant 60 over the top 10 from each retriever. Reranker: the existing cross-encoder (`bge-reranker-v2-m3`) rescoring the 10 candidates, keep the top 5.

**Implementation note.** `Retriever.search()` currently always runs hybrid plus rerank (the mode flags were removed earlier to keep it simple), so R0–R3 need new code. Put it in an eval-only subclass in `src/evaluation/rag_eval/`, built from the existing stage methods, and leave the production `search()` unchanged:

| Config | Built from |
| --- | --- |
| R0 | `_vector_ranking` top 5 |
| R1 | `_bm25_ranking` top 5 |
| R2 | `_candidates` order, top 5 (no rerank) |
| R3 | `_vector_ranking` top 10, then `_rerank`, top 5 (`_candidates` always fuses BM25, so R3 needs its own path) |
| R4 | `search()` as it is |

Non-reranked configs have no reranker score; use the vector or RRF score in the `score` field, and note it.

**Procedure:**

1. Retrieval only, all 5 configs: compute part recall@5, P@5, F1@5, MRR, Hit@5 and retrieval latency. No LLM calls, so this takes minutes. Extend `evaluate.py`'s `MODES` dict, which was built for this.
2. Generate answers with R0 and with the config picked by the rule in step 3. Score with RAGAS and the correctness rubric.
3. **Write the selection rule down before running anything.** Suggested: carry R4 (hybrid + rerank) forward unless its part recall@5 is below R0's, or its median retrieval latency is more than 3× R0's with no recall gain; in that case carry the config with the highest part recall@5, breaking ties by MRR.
4. The selection uses the same questions as the final report, so treat the ablation as descriptive. Don't claim the chosen config is best in general. If R4 wins on retrieval but not on correctness, report that as a finding.

**Table 1a: retrieval quality** (answerable questions, n = ?)

| Config | Part recall@5 | P@5 | F1@5 | MRR | Hit@5 (all parts) | Retrieval latency (ms, median) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 Dense |  |  |  |  |  |  |
| R1 BM25 |  |  |  |  |  |  |
| R2 Hybrid |  |  |  |  |  |  |
| R3 Dense + rerank |  |  |  |  |  |  |
| R4 Hybrid + rerank |  |  |  |  |  |  |

**Table 1b: end-to-end answer quality** (naive pipeline, Qwen 3.5 4B)

| Config | Accuracy (%) | Mean correctness (0–2) | Numeric acc. (%) | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 |  |  |  |  |  |  |  |
| Best config |  |  |  |  |  |  |  |

### Stage 2: Naive vs agentic RAG (RQ3)

**Variable:** system (N1 naive, N2 naive with matched context, A1 agentic). **Fixed:** the Stage 1 retriever used by all three, the shared settings above, and Qwen 3.5 4B.

| System | Description | Why it is there |
| --- | --- | --- |
| N1 Naive | One retrieval, top 5 chunks, one generation with the shared answer function | The fixed-pipeline baseline the slides require |
| N2 Naive, matched context | Same as N1, but k set to A1's mean number of unique chunks seen (rounded up), or an equal context-token budget | Separates "more context" from "agentic behaviour" |
| A1 Agentic | The supervisor stack from `src/supervisor/`: decomposer (intake and query planning) → supervisor loop (search, decide) → auditor (filters chunks, suggests follow-up queries) → answer | System under test |

**What A1 is, and what N2 does not control.** A1 is the whole stack, not "retrieval as one tool". Beyond seeing more chunks, it also (a) drops chunks the auditor rejects, (b) adds planner-extracted user facts and assumptions to the answer prompt, and (c) can stop early or give up. N2 controls only the number of chunks. State this in the report. If time allows, add an **A1 without the auditor** run (auditor set to approve everything) to measure the auditor's own contribution.

**Running A1 on evaluation questions** (all through `handle_turn(user_id, messages, today=…)`):

- Use a **fresh `user_id` per question** (or delete the profile afterwards), otherwise profile facts from one question leak into the next.
- Pass a **fixed `today`** date for every run; otherwise "this year" questions drift with the clock.
- Point `UserStore(path=…)` at a **temporary directory**. By default it writes into `chroma_db/`, which is tracked by git, so evaluation would dirty the frozen corpus snapshot.
- Status `clarify`, `refused` or `no_answer` on an answerable question is scored as incorrect (Section 3.3).
- Register no tools, so web search and the tax calculator are off and both systems answer from the same corpus.

Run all three systems on every question, including out-of-scope ones. Run A1 three times (agent paths vary) and N1 and N2 once. Build N2 after A1's first run, once you know its mean chunks seen. Score all metrics from Section 3. For agentic retrieval metrics, use the union of chunks across calls and report chunks seen. Trajectory metrics (searches per question, queries skipped as near-duplicates, stop reason) go in the separate agent eval. Failed runs count as wrong (Section 3.3).

**Table 2a: overall**

| System | Accuracy (%) | Numeric acc. (%) | Refusal acc. | Faithfulness | Answer rel. | Ctx precision | Ctx recall | Part recall@5 | Chunks seen (mean) | Latency median / p90 (s) | Tokens per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 Naive |  |  |  |  |  |  |  |  | 5 |  |  |
| N2 Naive, matched context |  |  |  |  |  |  |  |  |  |  |  |
| A1 Agentic (mean of 3 runs) |  |  |  |  |  |  |  |  |  |  |  |

**Table 2b: accuracy by question type** (shows *where* agentic helps)

| Type (n) | N1 accuracy (%) | N2 accuracy (%) | A1 accuracy (%) | A1 − N2 (pp) |
| --- | --- | --- | --- | --- |
| definition (n = ) |  |  |  |  |
| number (n = ) |  |  |  |  |
| procedure (n = ) |  |  |  |  |
| multi-part (n = ) |  |  |  |  |
| edge-case (n = ) |  |  |  |  |

Add the same breakdown by source (online guide, REIT ETF guide, insurer guide) if space allows.

**Table 2c: paired outcomes** (A1 vs N2, per question, correctness 2 vs < 2; repeat for A1 vs N1)

|  | Agentic correct | Agentic not correct |
| --- | --- | --- |
| **Naive correct** |  |  |
| **Naive not correct** |  |  |

The off-diagonal cells are the questions that matter. They also feed the qualitative analysis.

**Table 2d: stratified by naive retrieval success** (answerable questions). Split the questions by whether N1 covered every part (Hit@5 = 1). Agentic RAG should help mainly when the first retrieval misses, so this split shows whether it does, even with 31 answerable questions.

| Stratum | n | N1 accuracy (%) | N2 accuracy (%) | A1 accuracy (%) |
| --- | --- | --- | --- | --- |
| N1 Hit@5 = 1 (retrieval succeeded) |  |  |  |  |
| N1 Hit@5 = 0 (retrieval missed) |  |  |  |  |

### Stage 3: Generator LLM comparison (RQ4)

**Variable:** the LLM that writes the answer (naive) or that decides and answers (agentic). **Fixed:** Stage 1 retriever, shared settings, prompts, judge.

**What a generator swap really changes.** Only the supervisor's `llm` callable can be replaced. The decomposer and the auditor call local Ollama directly (`INTAKE_MODEL` and `AUDITOR_MODEL`), so in an "agentic with Claude or GPT" run the API model makes the decide and answer calls while Qwen 3.5 4B still plans and audits. State this in the report. The supervisor passes an Ollama-style JSON schema to `llm(messages, schema)`, so each API model needs a small adapter (system message passed separately, schema mapped to the provider's structured-output option). If you want a fully API-driven agent, the decomposer and auditor would also need adapters, which is out of scope for this plan.

| Run | Generator | System |
| --- | --- | --- |
| G1 | Qwen 3.5 4B | Naive (reuse from Stage 2) |
| G2 | Qwen 3.5 4B | Agentic (reuse from Stage 2) |
| G3 | Anthropic model (e.g. a Haiku- or Sonnet-tier model) | Naive |
| G4 | Anthropic model | Agentic (supervisor LLM swapped) |
| G5 | OpenAI model (similar tier) | Naive |
| G6 | OpenAI model | Agentic (supervisor LLM swapped) |

This needs only 4 new runs, since G1 and G2 are reused. If time is short, run one API model only (G3 and G4) and add the other later. Pick API models of a similar price tier so the comparison is fair, and record exact model IDs. Count tool-call failures as wrong (Section 3.3).

**Table 3: generator comparison**

| Generator | System | Accuracy (%) | Numeric acc. (%) | Faithfulness | Answer rel. | Latency median (s) | Cost per 100 questions (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 3.5 4B | Naive |  |  |  |  |  | local |
| Qwen 3.5 4B | Agentic |  |  |  |  |  | local |
| Anthropic | Naive |  |  |  |  |  |  |
| Anthropic | Agentic (supervisor LLM only) |  |  |  |  |  |  |
| OpenAI | Naive |  |  |  |  |  |  |
| OpenAI | Agentic (supervisor LLM only) |  |  |  |  |  |  |

Key reading for RQ4: compare the agentic gain (agentic minus naive accuracy) per model. If the gain is largest for Qwen, agentic RAG compensates for a small model. If it is largest for API models, the small model can't drive the agent well.

### Run budget

| Stage | Generation runs | Judge calls (≈ 370 per run: 31 answerable questions × roughly 12 RAGAS and rubric calls) |
| --- | --- | --- |
| 1 | 2–3 (R0 + chosen config) | ~750–1,100 |
| 2 | 4 new (N2, A1 × 3); N1 reused from Stage 1 | ~1,500–1,900 (larger agentic contexts cost more) |
| 3 | 4 | ~1,500 |

Cache every generation (Section 9.3) so you never regenerate just to re-score.

**Compute budget.** An agent run makes several local LLM calls per question (intake, audit, decide, answer) plus one or more reranked searches. **Time 5 questions on the target machine before committing** to 3 repeats × 35 questions. The cluster script (`IS469_testing.sh`) has a 2-hour job limit and uses `--requeue`, so the runner must be resumable and each job must process a slice of the dataset (Section 9.3). Record which machine produced each run.

## 6. Qualitative analysis protocol

Read 10 cases in full, classify every failure in the test set with one taxonomy, and present 3 worked examples in the report.

### 6.1 Selecting the 10 cases

Select from Stage 2 and 3 outputs, using the paired table (Table 2c) and the score files:

| Slot | Count | How to find them |
| --- | --- | --- |
| Agentic correct, naive wrong | 3 | Off-diagonal cell of Table 2c; prefer different question types |
| Naive correct, agentic wrong | 2 | Other off-diagonal cell. These show where the agent hurts |
| Both wrong | 2 | Correctness 0 for both; reveals corpus or retrieval limits |
| Retrieval perfect (R@5 = 1) but answer wrong | 1 | Isolates generation failures from retrieval failures |
| Out-of-scope | 1 | One correct refusal or one failed refusal |
| Qwen wrong, API model right (Stage 3) | 1 | Shows what model size buys |

If a slot is empty, say so; that is itself a finding (e.g. "naive never beat agentic").

### 6.2 Failure taxonomy

Tag every answer scored below 2 with **one primary code** (the earliest failure in the pipeline). One member tags, a second checks 10.

| Code | Failure | How to tell |
| --- | --- | --- |
| F1 | Retrieval miss | No gold chunk in the retrieved set |
| F2 | Ranked too low | Gold chunk retrieved but outside the top 5 passed to the LLM (or below the reranker cutoff) |
| F3 | Partial context | Some parts covered, but not all (common on multi-part) |
| F4 | Unsupported claim | Answer states something not in the context (low faithfulness) |
| F5 | Wrong number | numeric\_ok = false while the correct number was in the context |
| F6 | Misread context | Right chunk, wrong conclusion (e.g. missed a condition or exception) |
| F7 | Incorrect refusal | Declined an answerable question (includes agent status `no_answer`) |
| F8 | Failed refusal | Answered an out-of-scope question |
| F9 | Incomplete answer | Correct but missing a required part |
| F10 | Run failure | Tool-call error, step-limit hit, timeout, empty answer, or an agent `clarify` or `refused` status on an answerable question |

F1–F3 are retrieval failures, F4–F6 and F9 are generation failures, F7–F8 are scope failures, and F10 is a run failure. Report a count table of codes per system. This tells you whether to fix retrieval or generation. For agentic runs, also check whether the **auditor** rejected a chunk that held a needed fact; code it F2 and note "auditor" in the diagnosis.

**Table 4: failure counts**

| Code | Naive (Qwen) | Agentic (Qwen) | Agentic (best API model) |
| --- | --- | --- | --- |
| F1 Retrieval miss |  |  |  |
| F2 Ranked too low |  |  |  |
| F3 Partial context |  |  |  |
| F4 Unsupported claim |  |  |  |
| F5 Wrong number |  |  |  |
| F6 Misread context |  |  |  |
| F7 Incorrect refusal |  |  |  |
| F8 Failed refusal |  |  |  |
| F9 Incomplete |  |  |  |
| F10 Run failure |  |  |  |

### 6.3 Worked-example template

```markdown
### Case Q__: <short title, e.g. "Agentic recovers a missed REIT ETF passage">
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

With 35 scored questions, one question moves accuracy by about 2.9 points. Report uncertainty on every headline number and compare systems per question, not just by averages.

1. **Report n everywhere.** Every table and per-category row shows its question count. Never report a percentage for a category with n < 5 without the raw count ("3/4").
2. **Bootstrap 95% confidence intervals** for accuracy and mean scores. Resample questions with replacement 1,000 times:

```python
import numpy as np
def bootstrap_ci(scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    means = [rng.choice(scores, len(scores)).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])
```

3. **Paired comparison for A1 vs N2 (and A1 vs N1).** Both systems answer the same questions, so compare per question: count wins, losses and ties (Table 2c). For a significance check on the binary "correct" outcome, use an exact McNemar test on the off-diagonal counts (`statsmodels.stats.contingency_tables.mcnemar(table, exact=True)`). It is optional: with about 45 questions it only reaches p < 0.05 when the off-diagonal cells are very lopsided (roughly 6–0 or 10–2), so the win/loss counts and the paired bootstrap in item 4 matter more.
4. **Paired bootstrap for the difference.** Resample question indices once per iteration and compute agentic minus naive on the same sample. If the 95% interval for the difference includes 0, write "no clear difference".
5. **Repeat non-deterministic runs.** Even at temperature 0, local models and API models can vary, and the agent's path can change. Run the agentic system (A1) 3 times and report mean ± range; run N1 and N2 once, since they are close to deterministic at temperature 0. If time only allows 1 run, say so as a limitation. Judge scores at temperature 0 are close to stable; re-run the judge once on 10 answers to confirm.
6. **Wording rules for the report:**

| Observed accuracy gap | How to describe it |
| --- | --- |
| ≤ 2 questions (≤ ~6 pp) | "comparable"; no claim of a winner |
| 3–5 questions, CI includes 0 | "slightly higher, but within noise" |
| CI excludes 0 and consistent across runs | "higher", with the interval |

7. **Don't over-read per-category results.** A category with 6 questions can swing 17 points on one answer. Use per-category tables to describe patterns and pick qualitative cases, not to claim significance.

Paraphrased questions share their answer with an original, so they are not independent observations. When you compute intervals or tests, resample by original question (a cluster bootstrap) so each original and its paraphrases count as one unit.

## 8. Trade-off discussion framework

Judge each alternative against the baseline on four dimensions, then state whether its quality gain justifies its extra cost. The slides' rule is that agentic RAG has to earn its extra cost and latency.

### 8.1 Dimensions

| Dimension | Measured by | Direction |
| --- | --- | --- |
| Quality | Accuracy, numeric accuracy, faithfulness | Higher is better |
| Latency | Median and p90 seconds per question | Lower is better; p90 matters for a chatbot user |
| Cost | Tokens per question; USD per 100 questions for API models | Lower is better |
| Complexity | Extra components, prompt/tool engineering, failure modes added (qualitative) | Lower is better |

### 8.2 Summary table for the report

Express every alternative relative to its baseline so trade-offs are visible in one row.

| Alternative vs baseline | Δ accuracy (pp, 95% CI) | Δ faithfulness | Latency ratio (×) | Token ratio (×) | Complexity added | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid vs dense |  |  |  |  | BM25 index |  |
| Rerank vs no rerank |  |  |  |  | Cross-encoder model |  |
| Agentic vs matched-context naive (Qwen) |  |  |  |  | Decomposer, auditor, supervisor loop |  |
| API model vs Qwen (agentic) |  |  |  |  | API dependency, cost |  |

### 8.3 Decision rule

Agree a rule *before* seeing results, so the verdict isn't chosen after the fact. Suggested:

- **Worth it:** accuracy gain over the matched-context baseline N2 whose 95% CI excludes 0, with latency ratio ≤ 3×.
- **Situational:** gain mainly on one question type (e.g. multi-part), or latency ratio > 3×. Recommend routing: use the cheaper system by default and the costlier one for that type.
- **Not worth it:** no clear accuracy gain (CI includes 0), or faithfulness drops.

Also report a simple efficiency figure: **extra correct answers per extra second** = (Δ correct answers) ÷ (Δ total latency over the test set). It turns quality vs latency into one comparable number.

For the discussion text, cover per alternative: what improved and on which question types, what it cost, and who should pick it (e.g. "for a tax chatbot where a wrong rate is costly, a 2× slower system with higher numeric accuracy is justified").

## 9. Implementation plan

Build one small evaluation harness that runs any config on the dataset, caches outputs to JSONL, and scores them separately. Generating and scoring as separate steps means a judge fix never forces a regeneration.

### 9.1 Codebase check: what exists and what to build

The plan was checked against the repo on 28 Oct 2026. The structure fits; these items must be built or decided first.

| # | Finding | Required action |
| --- | --- | --- |
| 1 | **The existing naive generator is not comparable.** `RAGGenerator` in `src/inference/inference.py` uses its own prompt, temperature 0.1, no `think=False` (a Qwen thinking model may think before answering) and passes whole chunks. The supervisor uses `ANSWER_SYSTEM`, temperature 0, `think=False`, and cuts each chunk to 1,200 characters (`DOC_CHARS`). Chunks can reach 512 tokens, so facts late in a chunk are invisible to the agent but not to the naive system | Write **one shared answer function** (supervisor prompt, temperature 0, `think=False`, same truncation) used by N1, N2 and A1. Don't use `RAGGenerator` for results |
| 2 | **"Agentic" is the whole supervisor stack,** including the decomposer, the auditor filter, planner-added user facts and a user profile store. `handle_turn` can also return `clarify` or `refused` | Describe A1 this way in the report; add the A1-without-auditor run if time allows. Handle statuses, fresh `user_id`, fixed `today` and a temporary `UserStore` path (Section 5, Stage 2) |
| 3 | **No dense-only, BM25-only or no-rerank modes.** `Retriever.search()` always fuses and reranks | Eval-only subclass built from the stage methods (Section 5, Stage 1). Leave production `search()` unchanged |
| 4 | **Only the supervisor's `llm` is swappable.** The decomposer and auditor call local Ollama directly | State the Stage 3 limitation; write one adapter per API model (Section 5, Stage 3) |
| 5 | **Logging is not exposed.** `SupervisorResult` has no chunk ids or token counts; the run state is internal | Wrap `retrieval.run` (record every returned chunk) and the `llm` callable; patch both `ollama.chat` and `ollama.Client.chat` to sum token counts (Section 9.3) |
| 6 | **Dataset v2 is written but not yet committed or labelled.** `src/evaluation/eval_questions.json` now has all tags, reference answers and key facts, but no `gold_passages` | Commit it, then run `label_gold.py` to add `gold_passages` (Section 2.4) |
| 7 | **The tracked database is at risk.** `.gitignore` has `# chroma_db/` commented out, so `chroma_db/` is committed, and `UserStore` writes into it | Use a temporary `UserStore` path during evaluation; record the `chroma_db/` commit as the corpus snapshot |
| 8 | **Dependencies.** `ragas`, a judge SDK (`anthropic`, `openai` or Google) and `statsmodels` are not in `requirements.txt`, which pins torch 2.14 and transformers 5.x. `scikit-learn` and PyYAML are already present | Run the scoring scripts in a **separate virtual environment** (they only read the JSONL outputs). Generation keeps the existing environment |
| 9 | **Compute.** The cluster script has a 2-hour limit and 1 GPU; a laptop CPU takes about 18 s per reranked search at pool 10 | Time 5 questions first; make `run.py` resumable and slice-able; record the machine in every run config |

What already works and should be reused:

| Reused piece | Where |
| --- | --- |
| Retrieval stages, RRF constant 60, reranker | `src/retrieval/retriever.py` |
| Fact-matching helpers `norm()` and `fact_ranks()`, `MODES` dict, per-question rank printout | `src/retrieval/evaluate.py` |
| Per-stage timings, injectable `llm`, `handle_turn(today=…)`, the agent builder `build_supervisor(...)` | `src/supervisor/supervisor.py`, `src/supervisor/harness.py` |
| Fixed Ollama keep-alive pinning (a pattern for wrapping `ollama.chat`) | `pin_ollama_keep_alive()` in `supervisor.py` |
| Resumable cluster job with `--requeue` | `IS469_testing.sh` |

### 9.2 Libraries

| Purpose | Library |
| --- | --- |
| Vector store, dense retrieval | `chromadb` (existing) |
| BM25 | `rank_bm25` (existing) |
| Reranker | `sentence-transformers` `CrossEncoder` (existing) |
| Local LLMs | `ollama` (existing) |
| Config files | `PyYAML` (existing) |
| Kappa | `scikit-learn` (existing) |
| RAGAS metrics | `ragas` (scoring environment only; pin the exact version) |
| Judge / API generators | Provider SDKs (`anthropic`, `openai`, `google-genai`), or `langchain` wrappers that RAGAS accepts (scoring environment) |
| Analysis | `pandas`, `numpy`, `statsmodels` (McNemar), `matplotlib` (scoring environment) |

### 9.3 Repo structure and logging

```text
src/evaluation/
  eval_questions.json         # dataset v2: questions, tags, reference answers, key facts, gold_passages
  rag_eval/
    analyze.py                # merges every score into the report tables: bootstrap CIs, paired tests, by type, strata
    db_check/
      facts.py                # loads the chunks from ChromaDB; snippet matching shared by the two scripts below
      check_dataset.py        # validates the dataset against the corpus (writes nothing)
      label_gold.py           # fills gold_passages from the verbatim facts (--write)
    rag_eval_retrieval/
      retrieval_modes.py      # ModeRetriever: R0-R4 built from Retriever's stage methods
      metrics.py              # part recall / precision / F1 / MRR / hit against gold_passages
      evaluate_rag.py         # Stage 1 retrieval-only scores -> results/retrieval/<mode>.json
    rag_eval_approach/
      answer.py               # the ONE shared answer step (supervisor prompt, temperature 0, think off, truncation)
      systems.py              # NaiveSystem, AgenticSystem: each .answer(question, qid, today) -> record
      llm.py                  # make_llm(provider, model): one llm(messages, schema) shape for every model
      llm_anthropic.py, llm_openai.py, llm_google.py   # API adapters (Stage 3 generators, judge)
      tracking.py             # token counting across every Ollama / API call
      run.py                  # generate: config + dataset -> results/outputs/<run_id>.jsonl (resumable)
      score_retrieval.py      # retrieval metrics on what each run retrieved (no LLM)
      score_judge.py          # correctness rubric and refusal checks -> results/scores/<run_id>_judge.jsonl
      score_ragas.py          # RAGAS -> results/scores/<run_id>_ragas.csv (separate scoring environment)
      results_io.py           # result paths and loaders
      configs/                # one YAML per run (s1_naive_r0_dense, n1_naive_r4, n2_naive_matched, a1_agentic_qwen,
                              #   g_naive_* / g_agentic_* for Stage 3) + judge.yaml
      prompts/                # judge_correctness_v1.txt, judge_refusal_v1.txt
      requirements-api.txt, requirements-scoring.txt
    results/                  # retrieval/, outputs/ (never edited by hand), scores/, report_assets/
```

Each line in `outputs/<run_id>.jsonl` holds one question's full record:

```json
{"run_id": "s2_agentic_qwen_r1", "qid": "Q07", "config_hash": "a1b2c3", "git_commit": "9f8e7d",
 "timestamp": "2026-11-03T14:02:11", "machine": "cluster-gpu", "generator": "qwen3.5:4b", "system": "agentic",
 "status": "answered",
 "retrieved_seen": [{"id": "web:ab12cd34ef:3", "headings": "...", "text": "...", "score": 0.83, "rank": 1}],
 "context_used": [{"id": "web:ab12cd34ef:3", "text": "<as shown in the prompt, truncated>"}],
 "answer": "...", "stop_reason": "auditor_sufficient",
 "latency_s": {"intake": 2.1, "retrieval": 3.4, "audit": 5.2, "decide": 1.0, "answer": 4.0, "total": 15.7},
 "tokens": {"input": 2410, "output": 180}, "error": null}
```

- `run_id` = config name + repeat number. Never overwrite an existing output file; a re-run gets a new ID.
- Save the config YAML and its hash with every run. Store prompt versions as files, not inline strings. Record the corpus snapshot (git commit of `chroma_db/`, scrape date) and the agent limits (`max_steps`, `max_queries`, `max_chunks`, `audit_min_score`).
- **Two chunk lists per record.** `retrieved_seen` is every chunk any search returned (used for retrieval metrics and chunks seen). `context_used` is the text actually placed in the answer prompt (used for RAGAS). For naive systems both are the same top-k.
- **Capturing what `SupervisorResult` does not expose:** wrap `RetrievalAgent.run` to append each returned chunk to `retrieved_seen`; wrap the supervisor's `llm` callable to record token counts; and patch **both** `ollama.chat` and `ollama.Client.chat` (the auditor and the existing naive generator call the client method) to add `prompt_eval_count` and `eval_count` to a per-question counter.
- `run.py` skips questions already in the output file and accepts `--start/--end` (or a slice list), so a crash or a 2-hour cluster job limit resumes instead of restarting.
- Log errors per question and keep going. Report error counts.
- Set seeds where possible and record hardware for local Qwen runs.

Minimal run loop:

```python
import json, os, time, hashlib, yaml
from datetime import date

def run(config_path, dataset_path, out_path, start=0, end=None):
    cfg = yaml.safe_load(open(config_path))
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    done = {json.loads(l)["qid"] for l in open(out_path)} if os.path.exists(out_path) else set()
    system = build_system(cfg)    # NaiveSystem | MatchedNaiveSystem | AgenticSystem, with retriever mode, k, pool, generator
    today = date.fromisoformat(cfg["today"])        # fixed for every run
    with open(out_path, "a") as f:
        for q in json.load(open(dataset_path))[start:end]:
            if q["qid"] in done:
                continue
            t0 = time.perf_counter()
            try:
                # agentic: fresh user_id per question, temp UserStore, handle_turn(user_id, msgs, today=today)
                res = system.answer(q["question"], qid=q["qid"], today=today)
                err = None
            except Exception as e:
                res, err = {}, repr(e)
            rec = {"run_id": cfg["run_id"], "qid": q["qid"], "config_hash": cfg_hash,
                   "total_s": time.perf_counter() - t0, "error": err, **res}
            f.write(json.dumps(rec) + "\n")
```

### 9.4 Team roles

| Member | Owns | Also does |
| --- | --- | --- |
| A | Dataset: flagged questions, paraphrases, gold labelling and audit; commit `eval_dataset_v2.json` | Second-checks failure codes |
| B | Harness: `run.py`, `answer.py`, `systems.py`, `retrieval_modes.py`, logging wrappers; Stage 1 retrieval ablation | Retrieval metrics script |
| C | Judge + RAGAS: scoring scripts and separate environment, judge validation, API adapters | Stage 3 API model runs |
| D | Stage 2 runs (agent wrapper: fresh user, fixed date, temp store), analysis script, statistics, qualitative analysis lead | Assembles report tables and figures |

Everyone completes about 4 of the flagged questions, audits about 4 gold labels, and scores about 15 judge-validation pairs (two scorers per pair).

### 9.5 Timeline (working back from 10 Nov)

Assumes the earlier G1 deadline. G2 (13 Nov) gains 3 days of buffer.

| Dates | Milestone | Owner | Done when |
| --- | --- | --- | --- |
| 27–29 Oct | Write the shared answer function; verify the 11 `needs_review` reference answers (dataset v2 is already written: 35 scored + 5 paraphrases); run `label_gold.py` and hand-check about 15 questions' labels; harness runs N1 and A1 end to end on 5 questions; time those 5 questions on the cluster | A, B, D | `eval_dataset_v2.json` committed and frozen; gold audit passed |
| 30 Oct – 1 Nov | Retrieval modes R0–R3; Stage 1 retrieval metrics for R0–R4; judge prompt written; Stage 1 generation (R0 + chosen config) | B; C | Table 1a filled; selection rule applied |
| 2–3 Nov | Judge validated on 30 pairs; Stage 2 runs: N1, A1 (3 repeats), then N2 matched to A1's mean chunks seen | C; D | Judge agreement ≥ 80%; Tables 1b, 2a–2d filled |
| 4–5 Nov | Stage 3 API model runs and scoring | C | Table 3 filled |
| 6–7 Nov | Failure coding (F1–F10), 10 qualitative cases, bootstrap CIs, trade-off table | D, A | Tables 4 and 5, 3 worked examples |
| 8–9 Nov | Write the evaluation section; final checks | All | Draft reviewed by 2 members |
| 10 Nov | Submit | All |  |

Set a hard cut-off: no new configs after 5 Nov. Anything later goes into the report as a limitation.

## 10. Deliverables and evaluation section outline

The report's RAG evaluation needs 6 tables, 3 figures and 3 worked examples.

### 10.1 Tables and figures

| ID | Content | From |
| --- | --- | --- |
| Table 0 | Dataset summary: counts by type, source, difficulty; judge–human agreement | Sections 2, 4.3 |
| Table 1a / 1b | Retrieval ablation: retrieval metrics (R0–R4), end-to-end quality (R0 vs best) | Stage 1 |
| Table 2a / 2b / 2c / 2d | N1 vs N2 vs A1: overall, by question type, paired outcomes, stratified by naive retrieval success | Stage 2 |
| Table 3 | Generator comparison | Stage 3 |
| Table 4 | Failure code counts per system | Section 6.2 |
| Table 5 | Trade-off summary with verdicts (Section 8.2) | Section 8 |
| Figure 1 | Part recall@5 and MRR per retriever config, bar chart with bootstrap CIs | Table 1a |
| Figure 2 | Accuracy by question type, N1 vs N2 vs A1, grouped bars | Table 2b |
| Figure 3 | Quality vs latency scatter: one point per system × generator, accuracy on y, median latency on x | Tables 2a, 3 |

Figure 3 shows the whole trade-off at a glance: points up and to the left are better.

### 10.2 Evaluation section outline

1. **Evaluation setup** (½ page): dataset and tags (Table 0), how gold passages were labelled, the shared settings table, what "agentic" means in this repo, the judge model and its validated agreement.
2. **Metrics** (½ page): one line per metric with why it fits a tax chatbot; why BLEU/ROUGE were not used.
3. **Stage 1: retrieval ablation** (½–1 page): Tables 1a and 1b, Figure 1. Which component helped retrieval, and whether it carried through to answers.
4. **Stage 2: naive vs agentic RAG** (1 page): Tables 2a–2d, Figure 2. Lead with the paired result and its CI.
5. **Stage 3: generator comparison** (½–1 page): Table 3. The agentic gain per model, and the supervisor-LLM-only caveat.
6. **Qualitative analysis** (1 page): Table 4 and 3 worked examples, chosen to explain the biggest quantitative findings.
7. **Trade-offs** (½ page): Table 5 and Figure 3, applying the decision rule from Section 8.3.

Open each subsection with its finding and number (e.g. "Agentic RAG answered 6 more questions correctly than naive RAG (+13 pp, 95% CI ...), mostly on multi-part questions"), then the table.

## Assumptions to confirm

These choices were assumed; change them if they don't match your setup.

- Dataset v2 (35 scored questions + 5 paraphrases) is written at `src/evaluation/eval_questions.json` and will be committed to the repo. 11 records are flagged `needs_review` and must be verified before the judge runs.
- Gold passages are auto-labelled with `label_gold.py` and audited by hand. The corpus is not re-ingested, so chunk ids stay valid.
- Final k = 5 for every system; reranker candidate pool 10 (the existing `CANDIDATES`); RRF constant 60.
- All systems share one answer function (supervisor prompt, temperature 0, thinking off, 1,200-character truncation).
- The agent runs with its default limits (`max_steps` 4, `max_queries` 2, `max_chunks` 6, `audit_min_score` 0.5), no tools, a fixed `today`, a fresh user per question and a temporary profile store.
- Agent statuses `clarify`, `refused` and `no_answer` on answerable questions count as incorrect.
- Qwen 3.5 4B runs through Ollama (laptop or the cluster GPU), so its cost is reported as tokens and latency, not USD. Record the machine per run.
- In Stage 3 only the supervisor's LLM is swapped; the decomposer and auditor stay on Qwen 3.5 4B.
- RAGAS class names follow recent releases (0.2+). Check them against the version you install, in the separate scoring environment.
- The deadline is 10 Nov (G1). With 13 Nov (G2), add the 3 days as buffer before writing.
- You have API access to a judge model outside the compared generators. If not, see Section 4.1.

### Not covered by this plan

Reflection, future work and risks/guardrails are deferred, but the course rubric still requires them. Beyond that, this plan does not cover:

- **Citation accuracy:** whether the answer cites the right source. For a tax chatbot this is worth adding if time allows.
- **Multi-turn conversations:** every question is single-turn. The agent supports clarification turns and user profiles; test a few separately if the chatbot will be used that way.
- **Time-sensitive answers:** results depend on the corpus snapshot (e.g. the 30 Nov 2026 deadline, the YA 2026 rebate), so record the scrape date.
- **Latency across local and API models:** not directly comparable, so compare naive vs agentic within each generator.
- **A fully API-driven agent:** the decomposer and auditor stay on local Qwen (Section 5, Stage 3).
