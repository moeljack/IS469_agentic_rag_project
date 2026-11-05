"""Rename question ids in saved files, using src/evaluation/qid_map.json (old id -> new id).

The ids were renumbered from Q01-Q64 (with gaps) to a scheme that says what a question is:
    S01-S21  single-fact answerable      M01-M20  multi-part answerable
    O01-O04  out-of-scope                P01-P05  paraphrases (retrieval only; `paraphrase_of` names the original)
    A01-A10  multi-step answerable (qtype "multi-step", added later; never had Q ids, so the map leaves them alone)
    Later: P01-P05 moved to agent_eval/data/robustness_cases.json, and O05-O20 were added (new ids, not in the map).
The map is the record of which old id became which new id.

Run:  python src/evaluation/rag_eval/db_check/apply_qid_map.py src/evaluation/rag_eval/results      rewrites in place
It touches every .json / .jsonl file under the folders you give, replacing only whole quoted ids ("Q22" -> "S10"),
so text inside answers is never changed. It is safe to run twice: new ids are never old ids. Use it on any results
folder produced before the renumbering (for example one copied back from the cluster).
"""
import json
import re
import sys
from pathlib import Path

MAP = json.loads((Path(__file__).resolve().parents[2] / "qid_map.json").read_text(encoding="utf-8"))
QUOTED = re.compile(r'"(Q\d\d)"')

changed = 0
for folder in sys.argv[1:]:
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix not in (".json", ".jsonl"):
            continue
        text = path.read_bytes().decode("utf-8")   # bytes, so line endings are kept
        new = QUOTED.sub(lambda m: f'"{MAP[m.group(1)]}"' if m.group(1) in MAP else m.group(0), text)
        if new != text:
            path.write_bytes(new.encode("utf-8"))
            changed += 1
print(f"{changed} files rewritten")
