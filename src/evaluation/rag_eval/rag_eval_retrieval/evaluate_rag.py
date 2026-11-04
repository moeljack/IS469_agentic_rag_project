"""Stage 1 of the evaluation plan: retrieval quality of R0-R4, with no LLM calls.

Each answerable question in eval_questions.json has `gold_passages`: one list of chunk ids per fact the answer needs
(written by db_check/label_gold.py). For every retrieval mode this script retrieves `--depth` chunks per question and
scores the top `--k` of them (plan Section 3.1):

    part recall@k   share of the question's facts covered by the top k
    precision@k     share of the top k that are gold chunks
    F1@k            harmonic mean of the two
    MRR             1 / rank of the first gold chunk, over the full depth
    hit@k           every fact covered (strict; multi-part questions are the hardest to pass)

Out-of-scope questions are skipped (they have no facts). Results per mode are saved to
results/retrieval/<mode>.json for analyze.py.

Run:  python src/evaluation/rag_eval/rag_eval_retrieval/evaluate_rag.py --mode all
      python src/evaluation/rag_eval/rag_eval_retrieval/evaluate_rag.py --mode dense
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root, so `src` is importable

from src.config import CANDIDATES, RERANK_MODEL, RERANK_QUANTIZE
from src.evaluation.rag_eval.rag_eval_retrieval.metrics import mean, part_ranks, retrieval_metrics
from src.evaluation.rag_eval.rag_eval_retrieval.retrieval_modes import MODES, RERANKED, ModeRetriever

QUESTIONS = Path(__file__).resolve().parents[2] / "eval_questions.json"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "retrieval"
METRICS = (("r", "Recall"), ("p", "Prec"), ("f1", "F1"), ("rr", "MRR"), ("hit", "Hit"))

parser = argparse.ArgumentParser(description="Score retrieval modes R0-R4 against the gold chunk ids")
parser.add_argument("--mode", default="hybrid_rerank", choices=[*MODES, "all"], help="retrieval mode, or all five")
parser.add_argument("--k", type=int, default=5, help="cutoff for recall, precision, F1 and hit (chunks given to the generator)")
parser.add_argument("--depth", type=int, default=10, help="how many chunks to retrieve (MRR and the rank table look this deep)")
parser.add_argument("--candidates", type=int, default=CANDIDATES, help="candidate pool for the fused and reranked modes")
parser.add_argument("--limit", type=int, help="score only the first N questions (smoke test)")
args = parser.parse_args()

questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
questions = [q for q in questions if q.get("answerable", True)]   # out-of-scope questions have no facts to find
questions = questions[:args.limit] if args.limit else questions
missing = [q["qid"] for q in questions if "gold_passages" not in q]
if missing:
    sys.exit(f"No gold_passages for {missing}. Run src/evaluation/rag_eval/db_check/label_gold.py --write first.")

modes = list(MODES) if args.mode == "all" else [args.mode]
retriever = ModeRetriever(modes[0])
if RERANKED & set(modes):
    retriever.mode = "hybrid_rerank"
    retriever.search("warm-up", k=1)   # load the reranker now so it isn't counted in the timings


def summarise(rows):
    out = {key: mean(r["metrics"][key] for r in rows) for key, _ in METRICS}
    out["n"] = len(rows)
    out["ms_median"] = statistics.median(r["seconds"] for r in rows) * 1000 if rows else float("nan")
    return out


def line(name, s):
    cells = "".join(f"{s[key]:>10.2f}" if key == "rr" else f"{s[key]:>10.0%}" for key, _ in METRICS)
    return f"{name:<22}{s['n']:>4}{cells}{s['ms_median']:>10.0f}"


header = f"{'mode':<22}{'n':>4}" + "".join(f"{label + '@' + str(args.k) if key != 'rr' else label:>10}"
                                          for key, label in METRICS) + f"{'ms (med)':>10}"
summaries = {}
for mode in modes:
    retriever.mode = mode
    rows = []
    began = time.perf_counter()
    for n, q in enumerate(questions, 1):
        if n % 5 == 0 or n == len(questions):   # reranked modes take ~20 s per query on CPU, so show progress
            spent = time.perf_counter() - began
            print(f"  {mode}: {n - 1}/{len(questions)} done, elapsed {spent / 60:.1f}m"
                  + (f", ETA {spent / (n - 1) * (len(questions) - n + 1) / 60:.1f}m" if n > 1 else ""), flush=True)
        start = time.perf_counter()
        hits = retriever.search(q["question"], k=args.depth, candidates=args.candidates)
        seconds = time.perf_counter() - start
        ids = [h["id"] for h in hits]
        rows.append({"qid": q["qid"], "type": q["type"], "qtype": q.get("qtype"), "source": q.get("source"),
                     "retrieved": [{"rank": i, "id": h["id"], "score": h["score"]} for i, h in enumerate(hits, 1)],
                     "part_ranks": part_ranks(ids, q["gold_passages"]),
                     "metrics": retrieval_metrics(ids, q["gold_passages"], k=args.k), "seconds": seconds})
    summary = {"all": summarise(rows), **{t: summarise([r for r in rows if r["type"] == t]) for t in ("simple", "multi")}}
    summaries[mode] = summary
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{mode}.json").write_text(json.dumps({
        "mode": mode, "label": MODES[mode], "k": args.k, "depth": args.depth, "candidates": args.candidates,
        "reranker": RERANK_MODEL if mode in RERANKED else None, "int8": RERANK_QUANTIZE,
        "summary": summary, "questions": rows}, indent=1), encoding="utf-8")

    print(f"\n{MODES[mode]} ({mode}) | k {args.k} | depth {args.depth} | pool {args.candidates} | "
          f"saved to {RESULTS / f'{mode}.json'}")
    print(header)
    for part in ("all", "simple", "multi"):
        if summary[part]["n"]:
            print(line(f"  {part}", summary[part]))
    print(f"\nRank of the first gold chunk for each fact ('-' = not in the top {args.depth})")
    for i, (q, r) in enumerate(zip(questions, rows), 1):
        ranks = ",".join(str(x or "-") for x in r["part_ranks"])
        print(f"{i:<4}{q['qid']:<5}{q['type'][:5]:<7}{(q.get('source') or '')[:7]:<8}{ranks:>8}   {q['question'][:60]}")

if len(modes) > 1:
    print(f"\nAll modes ({len(questions)} questions)")
    print(header)
    for mode in modes:
        print(line(MODES[mode], summaries[mode]["all"]))
