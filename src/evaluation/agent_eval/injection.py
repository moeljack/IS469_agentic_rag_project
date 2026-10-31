"""Plant a malicious chunk in retrieval results, to test whether the agent obeys instructions found in retrieved text.

In agentic RAG, retrieved text is untrusted input: an instruction can ride in on a chunk and try to hijack a tool-using
agent. InjectingRetriever wraps the evaluation's ModeRetriever and, while an injection case is active, adds the
attacker's text to what search() returns:

    standalone   a separate chunk with id "inj:<case_id>", inserted at rank 1 with reranker score 0.97, as if a poisoned
                 page had been ingested and matched the query well
    append       the attacker's text is appended to the top real chunk (a hidden instruction inside a relevant page);
                 that chunk's id is recorded in .poisoned_ids

The high score deliberately gets the chunk past the reranker cutoff (audit_min_score), so the test exercises the
defences that come after retrieval: the auditor, the answer prompt's rules, and the scoping of the agent's actions.
Everything else (models, collection, mode) is passed through to the wrapped retriever unchanged.
"""

INJECTED_SCORE = 0.97
DEFAULT_SOURCE = "https://www.iras.gov.sg/taxes/corporate-income-tax/notices"


class InjectingRetriever:
    def __init__(self, inner):
        self._inner = inner
        self.case = None              # the active injection case (dict from injection_cases.json), or None
        self.poisoned_ids = []        # ids of chunks that carried the attacker's text for the active case
        self.injections = 0           # how many search() calls returned the attacker's text

    def __getattr__(self, name):      # mode, collection, warm-up helpers... all from the real retriever
        return getattr(self._inner, name)

    def set_case(self, case):
        """Activate a case for the next question (None = clean retrieval)."""
        self.case = case if case and case.get("vector") == "chunk" else None
        self.poisoned_ids, self.injections = [], 0

    def search(self, query, k=5, **kwargs):
        hits = self._inner.search(query, k=k, **kwargs)
        case = self.case
        if case is None:
            return hits
        self.injections += 1
        if case.get("placement") == "append" and hits:
            top = dict(hits[0])
            top["text"] = f"{top['text']}\n{case['payload']}"
            if top["id"] not in self.poisoned_ids:
                self.poisoned_ids.append(top["id"])
            return [top] + hits[1:]
        chunk_id = f"inj:{case['case_id']}"
        if chunk_id not in self.poisoned_ids:
            self.poisoned_ids.append(chunk_id)
        injected = {"id": chunk_id, "text": case["payload"], "score": INJECTED_SCORE,
                    "source": case.get("source", DEFAULT_SOURCE), "headings": case.get("heading", ""),
                    "pages": "", "source_type": "web", "title": case.get("heading", "")}
        return ([injected] + [h for h in hits if h["id"] != chunk_id])[:max(k, 1)]
