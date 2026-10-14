"""Measure retrieval quality on eval_questions.json.

A question counts as found at rank r when the r-th returned chunk (heading + text, case and spacing
ignored) contains the question's `expect` snippet. Run:  python src/retrieval/evaluate.py
Add questions to eval_questions.json as you think of them; keep `expect` short and specific to the right chunk.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.retrieval.retriever import Retriever

QUESTIONS = Path(__file__).with_name("eval_questions.json")
MAX_RANK = 10   # how deep to look for the answer
MODES = {       # name -> Retriever.search options
    "vector":             {"hybrid": False, "rerank": False},
    "vector + rerank":    {"hybrid": False, "rerank": True},
    "hybrid":             {"hybrid": True,  "rerank": False},
    "hybrid + rerank":    {"hybrid": True,  "rerank": True},
}


def norm(text):
    return re.sub(r"\s+", " ", text).lower()


def first_hit_rank(hits, expect):
    for rank, h in enumerate(hits, 1):
        if norm(expect) in norm(f"{h['headings']} {h['text']}"):
            return rank
    return None   # not in the top MAX_RANK


questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
retriever = Retriever()
retriever.search("warm-up", k=1)   # load the reranker now so it isn't counted in the timings

ranks = {mode: [] for mode in MODES}
seconds = {mode: 0.0 for mode in MODES}
for q in questions:
    for mode, options in MODES.items():
        start = time.perf_counter()
        hits = retriever.search(q["question"], k=MAX_RANK, **options)
        seconds[mode] += time.perf_counter() - start
        ranks[mode].append(first_hit_rank(hits, q["expect"]))

n = len(questions)
print(f"\n{n} questions, looking {MAX_RANK} deep\n")
print(f"{'mode':<18}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}{'sec/query':>11}")
for mode, r in ranks.items():
    at = lambda k: sum(1 for x in r if x is not None and x <= k) / n
    mrr = sum(1 / x for x in r if x) / n
    print(f"{mode:<18}{at(1):>7.0%}{at(3):>7.0%}{at(5):>7.0%}{mrr:>8.2f}{seconds[mode] / n:>11.2f}")

print("\nRank of the expected chunk per question ('-' = not in top %d):" % MAX_RANK)
print(f"{'':<4}" + "".join(f"{m:>18}" for m in MODES) + "   question")
for i, q in enumerate(questions):
    cells = "".join(f"{(ranks[m][i] or '-'):>18}" for m in MODES)
    print(f"{i + 1:<4}{cells}   {q['question'][:60]}")
