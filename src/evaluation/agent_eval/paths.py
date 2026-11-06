"""Where agent_eval reads and writes. Every agent_eval script takes its folders from here, and none of them reads or
writes rag_eval's result folders.

    results/output/<run_id>.jsonl          full runs over the question set (a1_agentic_qwen_rep1, n1_naive, ...):
                                           run_suite.py --suite main. The scorers' --baseline and the cost, trajectory
                                           and task-success scorers read these.
    results/<suite>/                       robustness, safety, toolcalls: the suite runs (run_suite.py), with the
                                           robustness and safety scorers' tables and per-case scores beside them
    results/<scorer>/                      trajectory, task_success, tool_calls, consistency, cost_latency

Inputs that are not results stay where they are: the question set (src/evaluation/eval_questions.json) and the run
configs (rag_eval/rag_eval_approach/configs/, shared with rag_eval so both evaluate the same systems).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/evaluation/agent_eval
EVAL = HERE.parent                              # src/evaluation
ROOT = HERE.parents[2]                          # repo root

RESULTS = HERE / "results"
OUTPUT = RESULTS / "output"
DATASET = EVAL / "eval_questions.json"
CONFIGS = EVAL / "rag_eval" / "rag_eval_approach" / "configs"


def results_dir(name):
    """agent_eval/results/<name>/, created if missing."""
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(path=DATASET):
    return {q["qid"]: q for q in json.loads(Path(path).read_text(encoding="utf-8"))}


def base_name(run_id):
    """The config a run belongs to: a1_agentic_qwen_rep2 -> a1_agentic_qwen."""
    return re.sub(r"_rep\d+$", "", run_id)