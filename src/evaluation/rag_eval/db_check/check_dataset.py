"""Check the evaluation dataset against the chunks in ChromaDB. Writes nothing.

Run:  python src/evaluation/rag_eval/db_check/check_dataset.py [--data src/evaluation/eval_questions.json]

Checks: valid JSON with every required field; unique qids; single-fact questions have one fact and multi-part ones at
least two; out-of-scope questions have none; every fact is found verbatim in at least one chunk (and not in a suspiciously
large number); no multi-part question has all its facts in one chunk; paraphrase_of points at a real question.
Prints the counts by type, question type, source and difficulty. Exit code 1 if there are errors.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import DATASET, load_chunks, matching_ids

REQUIRED = {"qid", "type", "tier", "qtype", "source", "difficulty", "numeric", "answerable", "question",
            "reference_answer", "key_facts", "facts", "needs_review"}
MAX_MATCHES = 5   # a fact matching more chunks than this is probably too generic to be a reliable label

parser = argparse.ArgumentParser(description="Check the evaluation dataset against the corpus")
parser.add_argument("--data", default=str(DATASET))
args = parser.parse_args()

data = json.loads(Path(args.data).read_text(encoding="utf-8"))
chunks = load_chunks()
errors, warnings = [], []

ids = [r["qid"] for r in data]
if len(ids) != len(set(ids)):
    errors.append(f"duplicate qids: {[k for k, v in Counter(ids).items() if v > 1]}")

for r in data:
    qid = r["qid"]
    if REQUIRED - set(r):
        errors.append(f"{qid}: missing fields {sorted(REQUIRED - set(r))}")
        continue
    if r["answerable"] and not r["facts"]:
        errors.append(f"{qid}: answerable but has no facts")
    if not r["answerable"] and r["facts"]:
        errors.append(f"{qid}: out-of-scope but has facts")
    if r["answerable"] and r["type"] == "simple" and len(r["facts"]) != 1:
        errors.append(f"{qid}: single-fact question should have exactly 1 fact")
    if r["answerable"] and r["type"] == "multi" and len(r["facts"]) < 2:
        errors.append(f"{qid}: multi-part question needs at least 2 facts")
    if "paraphrase_of" in r and r["paraphrase_of"] not in ids:
        errors.append(f"{qid}: paraphrase_of {r['paraphrase_of']} not found")

    matches = []
    for number, alternatives in enumerate(r["facts"], 1):
        found = matching_ids(chunks, alternatives)
        matches.append(set(found))
        if not found:
            errors.append(f"{qid} fact {number}: no alternative found in any chunk ({alternatives[0][:70]!r})")
        elif len(found) > MAX_MATCHES:
            warnings.append(f"{qid} fact {number}: matches {len(found)} chunks (too generic?)")
    if r["type"] == "multi" and len(matches) > 1 and all(matches):
        if set.intersection(*matches):
            errors.append(f"{qid}: all facts appear together in {sorted(set.intersection(*matches))}")
        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                if matches[i] == matches[j] and len(matches[i]) == 1:
                    errors.append(f"{qid}: facts {i + 1} and {j + 1} live in the same single chunk {sorted(matches[i])}")

scored = [r for r in data if "paraphrase_of" not in r]
answerable = [r for r in scored if r.get("answerable")]
print(f"records {len(data)} | scored {len(scored)} | answerable {len(answerable)} | "
      f"out-of-scope {len(scored) - len(answerable)} | paraphrases {len(data) - len(scored)}")
for key in ("type", "qtype", "source", "difficulty"):
    print(f"{key:<10}:", dict(Counter(r.get(key) for r in answerable)))
print(f"numeric   : {sum(bool(r.get('numeric')) for r in answerable)} | needs_review: {sum(bool(r.get('needs_review')) for r in data)}")
for w in warnings:
    print("WARN ", w)
for e in errors:
    print("ERROR", e)
print("OK" if not errors else f"{len(errors)} error(s)")
sys.exit(1 if errors else 0)
