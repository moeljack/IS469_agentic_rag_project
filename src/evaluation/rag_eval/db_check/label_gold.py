"""Label the gold chunks for every question: the ids of the chunks that contain each verbatim fact.

Run:  python src/evaluation/rag_eval/db_check/label_gold.py            report only (nothing is written)
      python src/evaluation/rag_eval/db_check/label_gold.py --write    add `gold_passages` to every record in the dataset file

`gold_passages` holds one list of chunk ids per fact, in the same order as `facts`. Those ids drive the retrieval
precision, recall and F1 scores. Chunk ids are positional (`web:<10 hex>:<n>`, `<pdf name>:<n>`), so re-run this after any
re-ingest. Facts that match no chunk, or more than 5, are flagged for a hand fix. Out-of-scope questions get an empty list.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facts import DATASET, load_chunks, matching_ids

MAX_MATCHES = 5

parser = argparse.ArgumentParser(description="Add gold chunk ids to the evaluation dataset")
parser.add_argument("--data", default=str(DATASET))
parser.add_argument("--write", action="store_true", help="save gold_passages into the dataset file")
args = parser.parse_args()

path = Path(args.data)
data = json.loads(path.read_text(encoding="utf-8"))
chunks = load_chunks()

flagged = 0
labelled = []
for record in data:
    gold = [matching_ids(chunks, alternatives) for alternatives in record["facts"]]
    for number, found in enumerate(gold, 1):
        if not found or len(found) > MAX_MATCHES:
            flagged += 1
            problem = "no chunk matches" if not found else f"{len(found)} chunks match (too generic?)"
            print(f"FLAG {record['qid']} fact {number}: {problem}: {record['facts'][number - 1][0][:70]!r}")
    new = {}
    for key, value in record.items():       # keep the field order: gold_passages right after facts
        if key != "gold_passages":
            new[key] = value
        if key == "facts":
            new["gold_passages"] = gold
    labelled.append(new)

facts_total = sum(len(r["facts"]) for r in data)
print(f"{len(data)} records, {facts_total} facts, {flagged} flagged")
if args.write:
    path.write_text(json.dumps(labelled, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"gold_passages written to {path}")
else:
    print("dry run: pass --write to save")
sys.exit(1 if flagged else 0)
