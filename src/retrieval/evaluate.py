"""Measure retrieval quality on eval_questions.json.

Each question has `expect` (a snippet that the right chunk contains) and `tier` ("webpage" or "pdf": where the
answer lives; "pdf" if omitted). A question counts as found at rank r when the r-th returned chunk (heading + text,
case and spacing ignored) contains the snippet. Run:  python src/retrieval/evaluate.py
Add questions to eval_questions.json as you think of them; keep `expect` short and specific to the right chunk.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.config import WEB_FIRST_MIN_SCORE
from src.retrieval.retriever import Retriever

QUESTIONS = Path(__file__).with_name("eval_questions.json")
MAX_RANK = 10   # how deep to look for the answer
MODES = {       # name -> how to search (search_tiered = web pages first, PDFs only as a fallback)
    "vector":          lambda r, q: r.search(q, k=MAX_RANK, hybrid=False, rerank=False),
    "hybrid":          lambda r, q: r.search(q, k=MAX_RANK, hybrid=True, rerank=False),
    "hybrid + rerank": lambda r, q: r.search(q, k=MAX_RANK, hybrid=True, rerank=True),
    "web first":       lambda r, q: r.search_tiered(q, k=MAX_RANK),
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
first_tier = []          # source type of the top hit of "web first", to check the routing
web_best = []            # best web-page reranker score per question, to calibrate the threshold
for q in questions:
    for mode, search in MODES.items():
        start = time.perf_counter()
        hits = search(retriever, q["question"])
        seconds[mode] += time.perf_counter() - start
        ranks[mode].append(first_hit_rank(hits, q["expect"]))
        if mode == "web first":
            first_tier.append(hits[0]["source_type"] if hits else None)
    top_web = retriever.search(q["question"], k=1, source_type="webpage")
    web_best.append(top_web[0]["score"] if top_web else 0.0)

n = len(questions)
tiers = [q.get("tier", "pdf") for q in questions]
print(f"\n{n} questions ({tiers.count('webpage')} answered by web pages, {tiers.count('pdf')} by PDFs), looking {MAX_RANK} deep\n")
print(f"{'mode':<18}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}{'sec/query':>11}")
for mode, r in ranks.items():
    at = lambda k: sum(1 for x in r if x is not None and x <= k) / n
    mrr = sum(1 / x for x in r if x) / n
    print(f"{mode:<18}{at(1):>7.0%}{at(3):>7.0%}{at(5):>7.0%}{mrr:>8.2f}{seconds[mode] / n:>11.2f}")

print(f"\nWeb-first routing (threshold {WEB_FIRST_MIN_SCORE}): the top hit should come from the question's own tier")
for tier in ("webpage", "pdf"):
    idx = [i for i, t in enumerate(tiers) if t == tier]
    ok = sum(first_tier[i] == tier for i in idx)
    scores = sorted(web_best[i] for i in idx)
    print(f"  {tier:<8} questions: {ok}/{len(idx)} routed correctly | best web-page score: "
          f"min {scores[0]:.2f}, median {scores[len(scores) // 2]:.2f}, max {scores[-1]:.2f}")
web_scores = [web_best[i] for i, t in enumerate(tiers) if t == "webpage"]
pdf_scores = [web_best[i] for i, t in enumerate(tiers) if t == "pdf"]
if min(web_scores) > max(pdf_scores):
    print(f"  clean separation: any threshold between {max(pdf_scores):.2f} and {min(web_scores):.2f} routes every question correctly")
else:
    print(f"  overlap: web questions as low as {min(web_scores):.2f}, PDF questions as high as {max(pdf_scores):.2f}")

print("\nRank of the expected chunk per question ('-' = not in top %d):" % MAX_RANK)
print(f"{'':<4}{'tier':<5}" + "".join(f"{m:>17}" for m in MODES) + f"{'web score':>11}   question")
for i, q in enumerate(questions):
    cells = "".join(f"{(ranks[m][i] or '-'):>17}" for m in MODES)
    print(f"{i + 1:<4}{tiers[i][:3]:<5}{cells}{web_best[i]:>11.2f}   {q['question'][:50]}")
