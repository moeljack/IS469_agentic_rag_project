"""The systems compared in Stages 1-3, each with .answer(question, qid, today) -> one output record.

    naive     one retrieval (any of R0-R4), top k chunks, one answer with the shared answer step   (N1, and N2 with k
              set to the agent's mean context size)
    agentic   the supervisor stack from src/supervisor (decomposer -> supervisor loop -> auditor -> answer)   (A1)

Every record has the same fields, so the scorers never need to know which system produced it:
    status            answered | no_answer | clarify | refused (the agent can also clarify or refuse)
    answer            the answer text (the agent's draft, without the Assumptions/Sources footer)
    retrieved_seen    every chunk a retrieval step returned, in order of first appearance (retrieval metrics)
    context_used      the chunks shown to the answer step, as truncated there (RAGAS contexts)
    latency_s         seconds per stage plus total
    tokens            LLM input/output tokens and calls, over every LLM call for the question

Running the agent on evaluation questions (plan Section 5, Stage 2): a fresh user id per question, a fixed `today`,
a temporary user store (the default one writes into the tracked chroma_db/), and no tools. SupervisorResult does not
expose which chunks were seen or used, so the retrieval step and the answer-prompt builder are wrapped to record them.
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "supervisor"))

from src.evaluation.rag_eval.rag_eval_approach.answer import context_used, generate_answer
from src.evaluation.rag_eval.rag_eval_approach.llm import make_llm
from src.evaluation.rag_eval.rag_eval_approach.tracking import USAGE, track_ollama
from src.evaluation.rag_eval.rag_eval_retrieval.retrieval_modes import RERANKED, ModeRetriever


def _seen(chunk_id, rank, score, source, headings):
    return {"rank": rank, "id": chunk_id, "score": score, "source": source, "headings": headings}


class NaiveSystem:
    name = "naive"

    def __init__(self, cfg, retriever):
        self.retriever, self.k, self.candidates = retriever, cfg["retriever"]["k"], cfg["retriever"]["candidates"]
        generator = cfg["generator"]
        self.llm = make_llm(generator["provider"], generator["model"], **generator.get("options", {}))
        self.retriever.search("warm-up", k=1, candidates=self.candidates)   # load the models before the first question

    def answer(self, question, qid, today):
        USAGE.reset()
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.k, candidates=self.candidates)
        t1 = time.perf_counter()
        draft, refused = generate_answer(self.llm, question, hits)
        t2 = time.perf_counter()
        return {"system": self.name, "status": "no_answer" if refused else "answered", "answer": draft,
                "retrieved_seen": [_seen(h["id"], i, h["score"], h["source"], h["headings"]) for i, h in enumerate(hits, 1)],
                "context_used": context_used(hits), "searches": 1,
                "latency_s": {"retrieval": t1 - t0, "generation": t2 - t1, "total": t2 - t0},
                "tokens": USAGE.snapshot()}

    def close(self):
        pass


class AgenticSystem:
    name = "agentic"

    def __init__(self, cfg, retriever, run_id):
        import supervisor
        from harness import build_supervisor
        from src.retrieval.user_store import UserStore

        agent = cfg["agent"]
        if retriever.mode not in RERANKED and agent["audit_min_score"]:
            raise ValueError(f"audit_min_score {agent['audit_min_score']} compares reranker scores (0-1), but mode "
                             f"{retriever.mode!r} has no reranker: set agent.audit_min_score to 0 for this mode")
        self.run_id, self.doc_chars = run_id, supervisor.DOC_CHARS
        self.store_dir = tempfile.mkdtemp(prefix="eval_user_store_")
        track_ollama()
        supervisor.pin_ollama_keep_alive()
        self.sup = build_supervisor(max_steps=agent["max_steps"], tools=None, store=UserStore(path=self.store_dir),
                                    k=cfg["retriever"]["k"], max_chunks=agent["max_chunks"],
                                    max_queries=agent["max_queries"], candidates=cfg["retriever"]["candidates"],
                                    audit_min_score=agent["audit_min_score"], retriever=retriever)
        generator = cfg["generator"]   # the supervisor's decide + answer calls; the decomposer and auditor stay on Ollama
        self.sup.llm = make_llm(generator["provider"], generator["model"], **generator.get("options", {}))

        run_search, build_answer_prompt = self.sup.retrieval.run, self.sup._answer_messages

        def recording_search(plan, queries=None, exclude=()):
            batch = run_search(plan, queries=queries, exclude=exclude)
            self._queries.append([q.text for q in (queries if queries is not None else plan.queries)])
            for c in batch:
                if c.id not in self._seen_ids:
                    self._seen_ids.add(c.id)
                    self._seen.append(_seen(c.id, len(self._seen) + 1, c.metadata.get("score"),
                                            c.metadata.get("source", ""), c.metadata.get("headings", "")))
            return batch

        def recording_answer_prompt(question, plan, docs, assumptions, partial):
            self._context = [{"id": d.id, "text": d.content[:self.doc_chars]} for d in docs]
            return build_answer_prompt(question, plan, docs, assumptions, partial)

        self.sup.retrieval.run = recording_search
        self.sup._answer_messages = recording_answer_prompt   # instance attribute: called without self, like the staticmethod
        self.sup.retrieval.warm()

    def answer(self, question, qid, today):
        self._seen, self._seen_ids, self._context, self._queries = [], set(), [], []
        USAGE.reset()
        result = self.sup.handle_turn(f"eval-{self.run_id}-{qid}", [{"role": "user", "content": question}], today=today)
        latency = {}
        for stage, seconds in result.timings:
            latency[stage] = latency.get(stage, 0.0) + seconds
        return {"system": self.name, "status": result.status,
                "answer": result.draft if result.status == "answered" else result.message,
                "stop_reason": result.stop_reason, "retrieved_seen": self._seen, "context_used": self._context,
                "searches": len(self._queries), "queries": self._queries, "latency_s": latency,
                "tokens": USAGE.snapshot(), "trace": result.trace}

    def close(self):
        shutil.rmtree(self.store_dir, ignore_errors=True)


def build_system(cfg, run_id):
    retriever = ModeRetriever(cfg["retriever"]["mode"])
    if cfg["system"] == "naive":
        return NaiveSystem(cfg, retriever)
    if cfg["system"] == "agentic":
        return AgenticSystem(cfg, retriever, run_id)
    raise ValueError(f"unknown system {cfg['system']!r}; choose naive or agentic")
