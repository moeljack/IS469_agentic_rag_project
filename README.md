# IS469_agentic_rag_project

## Setup (Windows)

Requires **Python 3.12** (3.11 also works). PyTorch does not publish wheels for very new Python versions.

```powershell
py -3.12 -m venv projenv
.\projenv\Scripts\Activate.ps1
python --version                     # should say 3.12.x
pip install -r requirements.txt
```

```

```

## Ingest documents

1. Put PDFs in `data/manual/`.
2. Run:
   ```powershell
   python src/ingest/ingest.py
   ```

On the **first run** the script downloads the models it needs into `models/`
(Docling layout/table models and `Qwen/Qwen3-Embedding-0.6B`, about 1.2 GB in total).
Later runs reuse them. To download ahead of time: `python src/ingest/download_models.py`.

What it does: PDF -> Docling -> chunks -> embeddings -> ChromaDB (`chroma_db/`, collection `manuals`).

| Command | Effect |
|---|---|
| `python src/ingest/ingest.py` | Ingest new PDFs, skip ones already stored |
| `python src/ingest/ingest.py --force` | Re-ingest every PDF |
| `python src/ingest/ingest.py --reset` | Wipe the collection and rebuild (required after changing `EMBED_MODEL`) |

### Scraped web pages

Put the scraper output (JSON lines with `url`, `title`, `text`, `retrieved_at`, `source`, `source_type: "webpage"`) in `data/raw/`, then run:

```powershell
python src/ingest/ingest_web.py            # new or re-scraped pages only (--force redoes everything)
```

Exact duplicate pages are merged, text is cleaned, and each page is split by section and table into chunks stored in
the same collection as the PDFs (`source` = the page URL, `source_type` = `webpage`).

## Searching

Edit the settings at the top of `src/retrieval/query.py`, then run `python src/retrieval/query.py`.
Each question is matched against all chunks (web pages and PDFs together) by vector and keyword search, the best
candidates are re-scored by a long-context reranker (`BAAI/bge-reranker-v2-m3`), and the top chunks are returned
with their `source_type`. Environment variables `RERANK_MODEL`, `RERANK_QUANTIZE=1` and
`CANDIDATES` change the reranker, switch it to faster int8 weights, and set the candidate pool size.
`python src/retrieval/evaluate.py` scores retrieval on `src/retrieval/eval_questions.json` (add your own
questions there).

Measured speed options for the reranker on the 40-question set (one search, CPU; accuracy figures are
simple-question hit@1 / hit@5 and multi-hop all-facts@5):

| Setting | Seconds per search | hit@1 | hit@5 | all@5 |
|---|---|---|---|---|
| default (full precision, pool 10) | 17.6 | 95% | 100% | 95% |
| `RERANK_QUANTIZE=1` | 11.9 | 85% | 100% | 95% |
| `CANDIDATES=6` | 10.5 | 90% | 95% | 95% |
| both | 6.6 | 85% | 95% | 95% |

The differences are one or two questions each, so treat them as indications. The default favours accuracy.

Inspecting with a GUI: after ingest finishes, run `chroma run --path chroma_db` and connect the GUI to
`http://localhost:8000`. **Stop the server (Ctrl+C) before running `ingest.py` or `inspect_db.py`**: two processes
using `chroma_db/` at the same time can lock or corrupt it.

Notes:
- OCR is off, so scanned (image-only) PDFs produce no text and are reported as failed.
- One failing PDF does not stop the run; failures are listed at the end.
- `models/`, `chroma_db/` and the virtual environment are git-ignored.

## Troubleshooting

- `WinError 1314` (symlink privilege) from Hugging Face: turn on Windows Developer Mode
  (Settings > System > For developers). The `models/` download avoids this, but other
  Hugging Face downloads may still hit it.
- `No matching distribution found for torch`: the venv uses an unsupported Python version; recreate it with 3.12.

## Agent interfaces (for mocking while agents are unfinished)

Each agent is described by what to call, what goes in and what comes out, so the supervisor can be built against
mock data. Only the Retrieval agent is filled in; owners complete the others in the same format.

### Harness
- **Call:** python3 src/supervisor/harness.py --user <id> --trace

### Supervisor agent
- **Call:** TODO
- **Input:** TODO
- **Output:** TODO

### Decomposer agent
- **Call:** TODO (file, function)
- **Input:** TODO
- **Output:** TODO

### Retrieval agent
Finds the chunks (web pages and PDFs, ranked together) that best answer **one** question. Returns chunks, not an answer.

**Call** - `src/retrieval/retriever.py`, `Retriever.search()`. Run from the project root.
```python
from src.retrieval.retriever import Retriever

retriever = Retriever()   # create ONCE and reuse: loads the models (first search also loads the reranker, ~25 s)
hits = retriever.search("What is the corporate tax rate in Singapore?", k=5)
```

**Input** - `Retriever.search(query, k=5, source=None, min_score=None, candidates=10)`

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `query` | `str` (required) | | One standalone question, no pronouns. One sub-question per call. |
| `k` | `int` | 5 | Maximum number of chunks to return. |
| `source` | `str` or `None` | `None` | Search only one document: a PDF filename or a page URL. |
| `min_score` | `float` or `None` | `None` | Drop hits whose `score` is below this (0 to 1). |
| `candidates` | `int` | 10 | Reranker pool size. Leave at the default. |

**Output** - returned by `Retriever.search()` itself (nothing is written to disk): a `list[dict]`, best first, at most `k`
items; an empty list if nothing passes `min_score`. Each dict:

| Key | Type | Meaning |
|---|---|---|
| `id` | `str` | Chunk id, e.g. `web:a5627a3f6e:1` or `<pdf-stem>:24`. |
| `score` | `float` | Reranker relevance, 0 to 1, higher is better. Comparable only within one query. |
| `text` | `str` | The chunk text (may contain Markdown tables). |
| `source` | `str` | Page URL (web) or PDF filename. Use for citations. |
| `source_type` | `"webpage"` or `"pdf"` | Where the chunk comes from. |
| `headings` | `str` | Section path such as `Payments > Credit Card Payment` (may be empty). |
| `pages` | `str` | PDF page numbers (`"9"` or `"9,10"`); empty for web pages. |
| `title` | `str` | Page title for web pages; empty for PDFs. |
| `vector_score`, `bm25_score` | `float` or `None` | First-stage scores, for diagnostics only. |

**Mock example** (what `hits` looks like):
```json
[
  {"id": "web:a5627a3f6e:1", "score": 0.993,
   "text": "We do not offer credit cards as a payment mode ...",
   "source": "https://www.iras.gov.sg/taxes/corporate-income-tax/corporate-income-tax-payment/paying-corporate-income-taxes",
   "source_type": "webpage", "headings": "Payments > Credit Card Payment", "pages": "", "title": "Payments",
   "vector_score": 0.68, "bm25_score": 28.25},
  {"id": "etaxguide_tax-deductibility-of-general-insurers-reserves-against-ibnr-claims_forth-edition:16", "score": 0.964,
   "text": "| Class of Business | Maximum IBNR Factor ... | Motor | 20 |",
   "source": "etaxguide_tax-deductibility-of-general-insurers-reserves-against-ibnr-claims_forth-edition.pdf",
   "source_type": "pdf", "headings": "Annex 1 - General Direct Insurance IBNR Percentage Table", "pages": "9", "title": "",
   "vector_score": 0.77, "bm25_score": null}
]
```

**Good to know:** a search takes about 10 to 20 s on CPU; neither source is preferred (chunks are ranked together by
`score`); hits carry no dates yet. To see real output quickly, edit `QUERIES` in `src/retrieval/query.py` and run it.

### Auditor agent
- **Call:** TODO (file, function)
- **Input:** TODO
- **Output:** TODO
