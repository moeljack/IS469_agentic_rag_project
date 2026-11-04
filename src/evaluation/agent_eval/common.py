"""Helpers shared by the agent-eval scorers: loading suite outputs, a judge-free key-fact check, part recall, tables."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.rag_eval.rag_eval_approach.results_io import OUTPUTS, RESULTS, read_jsonl  # noqa: E402

SUITE_OUT = Path(__file__).resolve().parent / "results"   # agent_eval/results/<suite>/ (run_suite.py writes here)
OLD_SUITE_OUT = RESULTS / "agent_eval"                     # where suite outputs went before; still read
ASSETS = RESULTS / "report_assets"

_MONTHS = {"january": "jan", "february": "feb", "march": "mar", "april": "apr", "june": "jun", "july": "jul",
           "august": "aug", "september": "sep", "sept": "sep", "october": "oct", "november": "nov", "december": "dec"}


def norm(text):
    """Lowercase, S$ -> $, full month names -> 3 letters, no thousands commas, single spaces."""
    t = str(text or "").lower().replace("s$", "$").replace("sgd ", "$")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    for full, short in _MONTHS.items():
        t = re.sub(rf"\b{full}\b", short, t)
    return re.sub(r"\s+", " ", t).strip()


def numeric_key_facts(q):
    """The key facts that carry a number (rates, amounts, dates); only these are checked without a judge."""
    return [k for k in (q.get("key_facts") or []) if re.search(r"\d", k)]


def key_fact_hit(answer, q):
    """Share of the question's numeric key facts found in the answer (None if it has none). A judge-free proxy for
    correctness: it can miss a correct paraphrase and cannot see a contradicting claim, so use it for comparisons
    between conditions, not as accuracy."""
    facts = numeric_key_facts(q)
    if not facts:
        return None
    a = norm(answer)
    return sum(1 for f in facts if norm(f) in a) / len(facts)


def part_recall(retrieved_ids, gold_parts):
    """(recall, hit): share of the question's facts with a gold chunk among everything retrieved, and all-covered."""
    if not gold_parts:
        return None, None
    ids = set(retrieved_ids)
    covered = [bool(ids & set(p)) for p in gold_parts]
    return sum(covered) / len(covered), int(all(covered))


def suite_files(suite):
    """{run_id: path} of a suite's run files, from agent_eval/results/<suite>/ and the old
    rag_eval/results/agent_eval/<suite>/. A run in both is read from the new folder."""
    found = {}
    for folder in (OLD_SUITE_OUT / suite, SUITE_OUT / suite):          # new folder last, so it wins
        for p in sorted(folder.glob("*.jsonl")):
            if not p.stem.endswith(("_scores", "_judge", "_tool_calls", "_task_success")):
                found[p.stem] = p
    return found


def suite_runs(suite, names=(), every=False):
    """{run_id: [records]} for a suite. A name without _repN takes all its repeats."""
    (SUITE_OUT / suite).mkdir(parents=True, exist_ok=True)              # the scorers write their scores here
    files = suite_files(suite)
    if not every:
        files = {r: p for r, p in files.items() if r in names or re.sub(r"_rep\d+$", "", r) in names}
        if not files:
            raise SystemExit(f"no {suite} outputs for {list(names)} in {SUITE_OUT / suite} or "
                             f"{OLD_SUITE_OUT / suite} (run run_suite.py first)")
    return {r: read_jsonl(p) for r, p in sorted(files.items())}


def suite_meta(suite, run_id):
    for folder in (SUITE_OUT / suite, OLD_SUITE_OUT / suite):
        path = folder / f"{run_id}.meta.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def baseline_records(name):
    """{qid: [records]} from results/outputs for a run id or a repeated config (all its _repN files)."""
    out = {}
    for p in sorted(OUTPUTS.glob("*.jsonl")):
        if p.stem == name or re.sub(r"_rep\d+$", "", p.stem) == name:
            for rec in read_jsonl(p):
                out.setdefault(rec["qid"], []).append(rec)
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rate(num, den):
    return num / den if den else None


def markdown(table):
    cols = list(dict.fromkeys(k for row in table for k in row))
    def cell(col, v):
        if v is None:
            return ""
        if isinstance(v, float):
            if v != v:
                return ""
            if any(w in col for w in ("rate", "share", "recall", "hit", "completion", "decline", "resilience",
                                      "utility")):
                return f"{v:.0%}"
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(c, row.get(c)) for c in cols) + " |" for row in table]
    return "\n".join(lines)


def save_table(name, table, extra=""):
    ASSETS.mkdir(parents=True, exist_ok=True)
    text = markdown(table) + (f"\n\n{extra}" if extra else "")
    (ASSETS / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    cols = list(dict.fromkeys(k for row in table for k in row))
    lines = [",".join(cols)] + [",".join(json.dumps(row[c]) if isinstance(row.get(c), str)
                                         else ("" if row.get(c) is None else str(row[c])) for c in cols)
                                for row in table]
    (ASSETS / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(text)
    print(f"\n  -> {ASSETS / (name + '.md')}")