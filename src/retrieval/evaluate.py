"""
Measure retrieval quality on eval_questions.json.

Each question has:
  type     "simple" (one fact in one chunk) or "multi" (two or more facts, each in a different chunk)
  tier     "webpage" or "pdf": where the answer lives
  facts    list of facts; each fact is a list of acceptable snippets (any one of them counts)

A fact is found at rank r when the r-th returned chunk (heading + text, case, spacing and apostrophe style ignored)
contains one of its snippets. A simple question is judged on its one fact (hit@k). A multi-hop question needs
all of its facts, so it is judged on how many are inside the top k (coverage@k) and whether all are (all@k).

Run:  python src/retrieval/evaluate.py      Add questions to eval_questions.json as you think of them.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable

from src.config import CANDIDATES, RERANK_MODEL, RERANK_QUANTIZE, ROOT
from src.retrieval.retriever import Retriever

QUESTIONS = Path(__file__).with_name("eval_questions.json")
MAX_RANK = 10   # how deep to look for the answer
KS = (1, 3, 5)
MODES = {       # name -> how to search (kept as a dict so saved hit files stay comparable with earlier runs)
    "hybrid + rerank": lambda r, q: r.search(q, k=MAX_RANK, candidates=POOL),
}


parser = argparse.ArgumentParser(description="Score retrieval on eval_questions.json")
parser.add_argument("--save", default=str(ROOT / "data" / "processed" / "eval_hits.json"),
                    help="where to write the retrieved chunk ids per question and mode, so a run can be re-scored later")
parser.add_argument("--k", type=int, default=MAX_RANK, help="how many chunks to retrieve (rank depth)")
parser.add_argument("--candidates", type=int, default=CANDIDATES, help="candidate pool size handed to the reranker")
args = parser.parse_args()
MAX_RANK, POOL = args.k, args.candidates


def norm(text):
    return re.sub(r"\s+", " ", text).lower().replace("’", "'")


def fact_ranks(hits, facts):
    """Rank (1-based) at which each fact first appears in the hits, or None if it is not in them."""
    haystacks = [norm(f"{h['headings']} {h['text']}") for h in hits]
    ranks = []
    for alternatives in facts:
        found = [i for i, text in enumerate(haystacks, 1) if any(norm(a) in text for a in alternatives)]
        ranks.append(found[0] if found else None)
    return ranks


questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
retriever = Retriever()
retriever.search("warm-up", k=1)   # load the reranker now so it isn't counted in the timings

ranks = {mode: [] for mode in MODES}      # per question: list of fact ranks
seconds = {mode: 0.0 for mode in MODES}
saved = []                                # per question: {mode: [{id, score, source_type}, ...]}
for qi, q in enumerate(questions):
    saved.append({})
    for mode, search in MODES.items():
        start = time.perf_counter()
        hits = search(retriever, q["question"])
        seconds[mode] += time.perf_counter() - start
        ranks[mode].append(fact_ranks(hits, q["facts"]))
        saved[qi][mode] = [{"id": h["id"], "score": h["score"], "source_type": h["source_type"]} for h in hits]

types = [q["type"] for q in questions]
tiers = [q["tier"] for q in questions]


def summarise(kind):
    idx = [i for i, t in enumerate(types) if t == kind]
    print(f"\n{kind.upper()} questions ({len(idx)})" + ("  [needs every fact in the top k]" if kind == "multi" else ""))
    if kind == "simple":
        print(f"{'mode':<18}" + "".join(f"{'hit@' + str(k):>8}" for k in KS) + f"{'MRR':>8}")
    else:
        print(f"{'mode':<18}" + "".join(f"{'all@' + str(k):>8}" for k in KS) + "".join(f"{'cov@' + str(k):>8}" for k in KS))
    for mode in MODES:
        rs = [ranks[mode][i] for i in idx]
        if kind == "simple":
            first = [r[0] for r in rs]
            cells = "".join(f"{sum(1 for x in first if x and x <= k) / len(rs):>8.0%}" for k in KS)
            mrr = sum(1 / x for x in first if x) / len(rs)
            print(f"{mode:<18}{cells}{mrr:>8.2f}")
        else:
            all_k = "".join(f"{sum(1 for r in rs if all(x and x <= k for x in r)) / len(rs):>8.0%}" for k in KS)
            cov_k = "".join(f"{sum(sum(1 for x in r if x and x <= k) / len(r) for r in rs) / len(rs):>8.0%}" for k in KS)
            print(f"{mode:<18}{all_k}{cov_k}")


Path(args.save).parent.mkdir(parents=True, exist_ok=True)
Path(args.save).write_text(json.dumps({"reranker": RERANK_MODEL, "hits": saved}), encoding="utf-8")
print(f"reranker: {RERANK_MODEL} (int8: {RERANK_QUANTIZE}) | pool {POOL} | depth {MAX_RANK} | hits saved to {args.save}")
print(f"\n{len(questions)} questions: {types.count('simple')} simple + {types.count('multi')} multi-hop | "
      f"{tiers.count('webpage')} answered by web pages, {tiers.count('pdf')} by PDFs | looking {MAX_RANK} deep")
summarise("simple")
summarise("multi")
print("\nSeconds per query: " + ", ".join(f"{m} {seconds[m] / len(questions):.1f}" for m in MODES))

print(f"\nRank of each fact per question ('-' = not in top {MAX_RANK}; multi-hop shows one rank per fact)")
print(f"{'':<4}{'type':<7}{'tier':<5}" + "".join(f"{m:>17}" for m in MODES) + "   question")
for i, q in enumerate(questions):
    cells = "".join(f"{','.join(str(x or '-') for x in ranks[m][i]):>17}" for m in MODES)
    print(f"{i + 1:<4}{types[i][:5]:<7}{tiers[i][:3]:<5}{cells}   {q['question'][:46]}")
