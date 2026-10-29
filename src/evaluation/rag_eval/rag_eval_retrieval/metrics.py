"""Retrieval metrics against the dataset's gold chunk ids (evaluation plan, Section 3.1).

Gold is stored as parts: `gold_passages` holds one list of chunk ids per fact the answer needs. A part is covered when
any of its chunks was retrieved, so a duplicate or equivalent chunk is not penalised.
"""


def part_ranks(retrieved_ids, gold_parts):
    """Rank (1-based) of the first retrieved chunk of each part, or None if no chunk of that part was retrieved."""
    return [next((rank for rank, id_ in enumerate(retrieved_ids, 1) if id_ in set(part)), None) for part in gold_parts]


def retrieval_metrics(retrieved_ids, gold_parts, k=None):
    """Part recall, precision, F1, reciprocal rank and hit for one question.

    k: cutoff for recall, precision, F1 and hit (None = the whole list, used for the agent's union of chunks seen).
    The reciprocal rank looks at the whole list, so it still shows the reranker's effect beyond the cutoff.
    """
    top = retrieved_ids if k is None else retrieved_ids[:k]
    denominator = k if k is not None else len(top)
    gold_all = {c for part in gold_parts for c in part}
    covered = [bool(set(top) & set(part)) for part in gold_parts]
    r = sum(covered) / len(gold_parts)
    p = len(set(top) & gold_all) / denominator if denominator else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rr = next((1 / rank for rank, id_ in enumerate(retrieved_ids, 1) if id_ in gold_all), 0.0)
    return {"p": p, "r": r, "f1": f1, "rr": rr, "hit": int(all(covered))}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    i = (len(values) - 1) * q / 100
    lo, hi = int(i), min(int(i) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)
