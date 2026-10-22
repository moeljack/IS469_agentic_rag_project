"""User store: who the user is and what we have learned about them from chatting, kept in ChromaDB.

Chroma is used as a small document store here, not for similarity search: every record is fetched by id or by
metadata filter, and carries a constant dummy vector so no embedding model is loaded. One collection holds three
kinds of record, told apart by the `kind` metadata field:

    profile:<user>        one per user   the facts we reuse as defaults + company details
    pending:<user>        0 or 1         entities from a clarification we asked, until the user answers
    hist:<user>:<ts>:<id> many           append-only log: every fact the user stated, with the question it came from

What is a "default" (PERSISTENT_FIELDS) is a deliberate choice: facts about the *company* (year end, new or not,
tax residency) carry over to the next question; facts about *one question* (chargeable income, the YA asked about)
do not. They are still written to the history log.

Used by supervisor.py:  ensure_profile, profile_entities, remember, save_pending, pop_pending
Used by harness.py:     set_company_info, forget
"""
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))   # schemas.py lives in src/inference
from schemas import Entities

COLLECTION = "user_store"
PERSISTENT_FIELDS = ("fye", "is_new_company", "incorporation_year", "tax_resident")   # reused as defaults
HISTORY_ONLY_FIELDS = ("ya", "fy", "chargeable_income")                                # logged, never reused
COMPANY_FIELDS = ("company_name", "uen")
_DUMMY_VECTOR = [1.0]          # Chroma wants an embedding per record; we never query by similarity


class UserStore:
    def __init__(self, client=None, collection_name: str = COLLECTION, path: Optional[str] = None):
        """client: any chromadb client (tests pass EphemeralClient). Default: a persistent client next to the
        vector DB (CHROMA_PATH from src.config, or `path`), in its own collection so it never mixes with chunks."""
        if client is None:
            import chromadb
            if path is None:
                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `src` importable
                from src.config import CHROMA_PATH
                path = str(CHROMA_PATH)
            client = chromadb.PersistentClient(path=path)
        self.col = client.get_or_create_collection(collection_name, embedding_function=None)

    # ------------------------------------------------------------ low-level record helpers
    def _get(self, id_: str) -> Optional[dict]:
        res = self.col.get(ids=[id_], include=["documents"])
        return json.loads(res["documents"][0]) if res["ids"] else None

    def _put(self, id_: str, user_id: str, kind: str, doc: dict, **meta) -> None:
        self.col.upsert(ids=[id_], embeddings=[_DUMMY_VECTOR], documents=[json.dumps(doc)],
                        metadatas=[{"user_id": user_id, "kind": kind, **meta}])

    # ------------------------------------------------------------ profile
    def get_profile(self, user_id: str) -> Optional[dict]:
        return self._get(f"profile:{user_id}")

    def ensure_profile(self, user_id: str) -> tuple[dict, bool]:
        """(profile, created). Created on the user's first message, then loaded."""
        profile = self.get_profile(user_id)
        if profile is not None:
            return profile, False
        profile = {"user_id": user_id, "created_at": time.time(), "company": {}, "facts": {}}
        self._put(f"profile:{user_id}", user_id, "profile", profile)
        return profile, True

    def set_company_info(self, user_id: str, **info) -> dict:
        unknown = set(info) - set(COMPANY_FIELDS)
        if unknown:
            raise ValueError(f"unknown company field(s): {sorted(unknown)}; allowed: {list(COMPANY_FIELDS)}")
        profile, _ = self.ensure_profile(user_id)
        profile["company"].update({k: v for k, v in info.items() if v})
        self._put(f"profile:{user_id}", user_id, "profile", profile)
        return profile

    def profile_entities(self, user_id: str) -> Entities:
        """The saved company facts as Entities, ready to be the planner's `prior`."""
        profile = self.get_profile(user_id)
        return Entities(**(profile["facts"] if profile else {}))

    # ------------------------------------------------------------ learning from the chat
    def remember(self, user_id: str, entities: Entities, question: str = "") -> list[tuple]:
        """Save what this turn taught us. Returns [(field, old, new)] for company facts that changed
        (the supervisor uses this to avoid claiming it 'reused' a value the user just gave)."""
        profile, _ = self.ensure_profile(user_id)
        changes = []
        for f in PERSISTENT_FIELDS:
            new = getattr(entities, f)
            old = profile["facts"].get(f)
            if new is not None and new != old:
                changes.append((f, old, new))
                profile["facts"][f] = new
        if changes:
            self._put(f"profile:{user_id}", user_id, "profile", profile)

        last = {}                                           # newest logged value per field, to skip exact repeats
        for h in self.history(user_id):
            last[h["field"]] = h["value"]
        for f in PERSISTENT_FIELDS + HISTORY_ONLY_FIELDS:
            v = getattr(entities, f)
            if v is None or (f in last and last[f] == v):
                continue
            ts = time.time_ns()
            self._put(f"hist:{user_id}:{ts}:{uuid.uuid4().hex[:6]}", user_id, "history",
                      {"field": f, "value": v, "question": question[:300], "ts": ts},
                      field=f, ts=ts)
        return changes

    def history(self, user_id: str, field: Optional[str] = None) -> list[dict]:
        """Everything the user has stated, oldest first. Optionally one field only."""
        where = {"$and": [{"user_id": user_id}, {"kind": "history"}] + ([{"field": field}] if field else [])}
        res = self.col.get(where=where, include=["documents"])
        return sorted((json.loads(d) for d in res["documents"]), key=lambda h: h["ts"])

    # ------------------------------------------------------------ clarification round trip
    def save_pending(self, user_id: str, entities: Entities) -> None:
        self._put(f"pending:{user_id}", user_id, "pending", entities.model_dump())

    def pop_pending(self, user_id: str) -> Optional[Entities]:
        """Entities saved when we asked a clarifying question; returned once, then deleted."""
        doc = self._get(f"pending:{user_id}")
        if doc is None:
            return None
        self.col.delete(ids=[f"pending:{user_id}"])
        return Entities(**doc)

    # ------------------------------------------------------------ privacy
    def forget(self, user_id: str) -> None:
        """Delete the profile, pending state and history of this user."""
        self.col.delete(where={"user_id": user_id})