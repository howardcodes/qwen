"""Core memory operating system implementation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import re
from typing import Protocol

from .config import settings
from .models import (
    Memory, MemoryStreamEntry, MemoryStreamKind, MemoryConflict, MemoryConflictResolution, MemoryEdge, MemoryStatus,
    MemoryType, RecallExplanation, RecallResult, RelationType, clamp_score, utc_now,
)
from .scoring import TOKEN_RE, hybrid_retrieval_score, keyword_score, quality_scores, tokenize
from .store import InMemoryStore


ACTIVE_ONLY_VECTOR_STATUSES = {MemoryStatus.ACTIVE}
INDEXABLE_MEMORY_TYPES = {MemoryType.USER_FACT, MemoryType.PREFERENCE, MemoryType.SEMANTIC, MemoryType.CONVERSATION_SUMMARY}


@dataclass(slots=True)
class MemoryCandidate:
    content: str
    type: MemoryType | str
    key: str | None = None
    value: str | None = None
    confidence: float = 0.75
    sensitivity: str = "low"
    source: str = "user_message"
    should_remember: bool = True
    reason: str = ""


@dataclass(slots=True)
class IngestionResult:
    action: str
    memory: Memory | None = None
    existing_memory: Memory | None = None
    conflict: MemoryConflict | None = None
    prompt: str | None = None


def memory_kind_for_type(memory_type: MemoryType | str) -> MemoryStreamKind:
    kind_map = {
        MemoryType.USER_FACT: MemoryStreamKind.FACT,
        MemoryType.PREFERENCE: MemoryStreamKind.PREFERENCE,
        MemoryType.SEMANTIC: MemoryStreamKind.FACT,
        MemoryType.CONVERSATION_SUMMARY: MemoryStreamKind.REFLECTION,
    }
    return kind_map.get(MemoryType(memory_type), MemoryStreamKind.OBSERVATION)

def stream_importance_1_to_10(content: str, memory_type: MemoryType | str = MemoryType.EPISODIC, confidence: float = 0.75) -> int:
    tokens = tokenize(content)
    score = 2
    lower = content.lower()
    if MemoryType(memory_type) in {MemoryType.USER_FACT, MemoryType.PREFERENCE, MemoryType.SEMANTIC, MemoryType.CONVERSATION_SUMMARY}:
        score += 3
    if any(marker in lower for marker in ["my name is", "likes", "prefer", "exam", "student", "asked about"]):
        score += 2
    if len(tokens) >= 6:
        score += 1
    if confidence >= 0.85:
        score += 1
    return max(1, min(10, score))

def should_index_memory(memory: Memory) -> bool:
    return memory.status == MemoryStatus.ACTIVE and (memory.importance_score >= 0.4 or memory.memory_type in INDEXABLE_MEMORY_TYPES or memory.metadata.get("stream_kind") in {"profile", "fact", "preference", "reflection"})


def memory_from_stream_entry(entry: MemoryStreamEntry) -> Memory:
    """Represent a memory-stream observation as an ephemeral recallable memory."""

    return Memory(
        id=entry.id,
        user_id=entry.user_id,
        content=entry.content,
        memory_type=MemoryType.EPISODIC,
        source_session=str(entry.metadata.get("source_session", "memory-stream")),
        confidence_score=0.70,
        importance_score=entry.importance_score / 10,
        novelty_score=0.5,
        stability_score=0.35,
        sensitivity=str(entry.metadata.get("sensitivity", "low")),
        tags={"memory-stream", entry.kind.value},
        metadata={**entry.metadata, "stream_entry_id": entry.id, "stream_kind": entry.kind.value},
        status=entry.status,
        created_at=entry.created_at,
        updated_at=entry.last_accessed_at,
        last_recalled_at=entry.last_accessed_at,
    )


class ConflictDetector(Protocol):
    def chat(self, messages: Sequence[object], *, model: str | None = None, temperature: float = 0.2, max_tokens: int | None = None) -> str: ...


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def upsert_memory(self, memory: Memory) -> None: ...
    def delete_memory(self, memory_id: str) -> None: ...
    def list_memory_ids(self) -> list[str]: ...


class MemoryOS:
    def __init__(self, store: InMemoryStore | None = None, *, embedding_provider: EmbeddingProvider | None = None,
                 require_live_embeddings: bool = False, fallback_embedding_dimensions: int = 1024,
                 conflict_detector: ConflictDetector | None = None, vector_index: VectorIndex | None = None) -> None:
        self.store = store or InMemoryStore()
        self.embedding_provider = embedding_provider
        self.require_live_embeddings = require_live_embeddings
        self.fallback_embedding_dimensions = fallback_embedding_dimensions
        self.conflict_detector = conflict_detector
        self.vector_index = vector_index

    def get_user_profile(self, user_id: str):
        return self.store.get_user_profile(user_id) if hasattr(self.store, "get_user_profile") else None

    def remember(self, *, user_id: str, content: str, memory_type: MemoryType | str = MemoryType.EPISODIC,
                 source_session: str, tags: Iterable[str] = (), confidence_score: float | None = None,
                 importance_score: float | None = None, novelty_score: float | None = None,
                 stability_score: float | None = None, metadata: dict[str, object] | None = None,
                 confidence_reasons: list[str] | None = None, status: MemoryStatus = MemoryStatus.ACTIVE,
                 sensitivity: str = "low", actor: str = "memory-agent") -> Memory:
        scores = quality_scores(content, self.store.list_memories(user_id))
        stream_importance = stream_importance_1_to_10(content, memory_type, confidence_score if confidence_score is not None else float(scores["confidence"]))
        memory = Memory(user_id=user_id, content=content, memory_type=memory_type, source_session=source_session,
            confidence_score=confidence_score if confidence_score is not None else float(scores["confidence"]),
            importance_score=importance_score if importance_score is not None else stream_importance / 10,
            novelty_score=novelty_score if novelty_score is not None else float(scores["novelty"]),
            stability_score=stability_score if stability_score is not None else float(scores["stability"]),
            tags=set(tags), metadata={**dict(metadata or {}), "stream_kind": memory_kind_for_type(memory_type).value, "stream_importance": stream_importance}, confidence_reasons=confidence_reasons or list(scores.get("confidence_reasons", [])),
            embedding=self._embed_text(content), status=status, sensitivity=sensitivity,
            approved_at=utc_now() if status == MemoryStatus.ACTIVE else None, last_confirmed_at=utc_now() if status == MemoryStatus.ACTIVE else None,
            last_seen_at=utc_now())
        if hasattr(self.store, "add_memory_stream_entry"):
            self.store.add_memory_stream_entry(MemoryStreamEntry(user_id=user_id, content=content, kind=memory_kind_for_type(memory_type), importance_score=stream_importance, metadata={"source_session": source_session, "memory_id": memory.id}), actor=actor)
        self.store.add_memory(memory, actor=actor)
        # Legacy semantic conflict handling for direct remember() calls. Structured ingestion uses confirmation instead.
        for old in self.store.list_memories(user_id):
            if old.id == memory.id or old.status != MemoryStatus.ACTIVE:
                continue
            if _subject_key(memory.content) and _subject_key(memory.content) == _subject_key(old.content) and keyword_score(memory.content, old) < 0.85:
                self.store.update_memory(old.id, actor="conflict-resolver", status=MemoryStatus.SUPERSEDED)
                self.store.add_edge(MemoryEdge(source_memory=memory.id, target_memory=old.id, relation_type=RelationType.CONTRADICTS))
                self.store.add_edge(MemoryEdge(source_memory=memory.id, target_memory=old.id, relation_type=RelationType.SUPERSEDES))
                self.sync_memory_vector(old)
        self.sync_memory_vector(memory)
        return memory

    def ingest_candidate(self, *, user_id: str, candidate: MemoryCandidate, source_session: str, actor: str = "memory-agent") -> IngestionResult:
        if not candidate.should_remember:
            return IngestionResult("skipped")
        content, key, value = normalize_candidate(candidate)
        if key and key.startswith("profile.") and hasattr(self.store, "upsert_user_profile"):
            profile_field = key.removeprefix("profile.")
            if profile_field in {"name", "age", "occupation", "timezone"} and candidate.confidence >= 0.70 and candidate.sensitivity in {"low", "medium"}:
                cast_value: object = int(value) if profile_field == "age" and str(value).isdigit() else value
                profile = self.store.upsert_user_profile(user_id, actor=actor, **{profile_field: cast_value})
                if hasattr(self.store, "add_memory_stream_entry"):
                    self.store.add_memory_stream_entry(MemoryStreamEntry(user_id=user_id, content=content, kind=MemoryStreamKind.PROFILE, importance_score=stream_importance_1_to_10(content, candidate.type, candidate.confidence), metadata={"source_session": source_session, "key": key, "value": value}), actor=actor)
                return IngestionResult("profile_updated")
        same_key = [m for m in self.store.list_memories(user_id) if m.memory_type == MemoryType(candidate.type) and m.metadata.get("key") == key]
        for old in same_key:
            if values_equal(str(old.metadata.get("value", "")), value):
                self.store.update_memory(old.id, actor=actor, last_seen_at=utc_now(), last_confirmed_at=utc_now(), confidence_score=max(old.confidence_score, candidate.confidence))
                self.sync_memory_vector(old)
                return IngestionResult("duplicate", memory=old, existing_memory=old)
        if same_key:
            existing = same_key[0]
            memory = self.remember(user_id=user_id, content=content, memory_type=candidate.type, source_session=source_session,
                tags={"agent", "extracted", "conflict"}, metadata={"source": candidate.source, "reason": candidate.reason, "key": key, "value": value},
                confidence_score=candidate.confidence, sensitivity=candidate.sensitivity, status=MemoryStatus.PENDING_CONFLICT_CONFIRMATION, actor=actor)
            memory.conflicting_memory_id = existing.id
            memory.conflict_reason = f"same key {key} has different value"
            self.store.update_memory(memory.id, actor=actor, conflicting_memory_id=existing.id, conflict_reason=memory.conflict_reason)
            conflict = self.store.add_conflict(MemoryConflict(user_id=user_id, existing_memory_id=existing.id, candidate_memory_id=memory.id,
                conflict_type=conflict_type_for_key(key), existing_content=existing.content, candidate_content=memory.content))
            self.store.add_edge(MemoryEdge(source_memory=memory.id, target_memory=existing.id, relation_type=RelationType.CONTRADICTS))
            return IngestionResult("conflict", memory=memory, existing_memory=existing, conflict=conflict, prompt=confirmation_prompt(existing, memory))
        lifecycle = policy_status_for_candidate(candidate, key)
        memory = self.remember(user_id=user_id, content=content, memory_type=candidate.type, source_session=source_session,
            tags={"agent", "extracted"}, metadata={"source": candidate.source, "reason": candidate.reason, "key": key, "value": value},
            confidence_score=candidate.confidence, sensitivity=candidate.sensitivity, status=lifecycle, actor=actor)
        return IngestionResult("created", memory=memory)

    def pending_conflict_for_user(self, user_id: str) -> MemoryConflict | None:
        return self.store.pending_conflict_for_user(user_id) if hasattr(self.store, "pending_conflict_for_user") else None

    def resolve_pending_conflict(self, user_id: str, reply: str, *, actor: str = "user") -> IngestionResult | None:
        conflict = self.pending_conflict_for_user(user_id)
        if not conflict:
            return None
        decision = classify_conflict_reply(reply, self.conflict_detector)
        existing = self.store.get_memory(conflict.existing_memory_id)
        candidate = self.store.get_memory(conflict.candidate_memory_id)
        if decision == "accept":
            self.store.update_memory(existing.id, actor=actor, status=MemoryStatus.SUPERSEDED)
            self.sync_memory_vector(existing)
            candidate = self.store.update_memory(candidate.id, actor=actor, status=MemoryStatus.ACTIVE, approved_at=utc_now(), last_confirmed_at=utc_now())
            self.sync_memory_vector(candidate)
            self.store.resolve_conflict(conflict.id, resolution=MemoryConflictResolution.ACCEPTED_CANDIDATE.value, actor=actor)
            return IngestionResult("accepted_candidate", memory=candidate, existing_memory=existing, conflict=conflict)
        if decision == "reject":
            candidate = self.store.update_memory(candidate.id, actor=actor, status=MemoryStatus.REJECTED)
            self.sync_memory_vector(candidate)
            self.store.resolve_conflict(conflict.id, resolution=MemoryConflictResolution.KEPT_EXISTING.value, actor=actor)
            return IngestionResult("kept_existing", memory=candidate, existing_memory=existing, conflict=conflict)
        self.store.update_memory(candidate.id, actor=actor, status=MemoryStatus.REJECTED)
        self.sync_memory_vector(candidate)
        self.store.resolve_conflict(conflict.id, resolution=MemoryConflictResolution.MERGED.value, actor=actor)
        return IngestionResult("clarification_needed", memory=candidate, existing_memory=existing, conflict=conflict)

    def recall(self, user_id: str, query: str, *, query_tags: Iterable[str] = (), limit: int | None = None, include_pending_review: bool = False) -> list[RecallResult]:
        limit = limit or settings.memory_recall_top_k
        results: list[RecallResult] = []
        query_embedding = self._embed_text(query)
        allowed = {MemoryStatus.ACTIVE} | ({MemoryStatus.PENDING_REVIEW} if include_pending_review else set())
        vector_candidates = self.store.vector_search(user_id, query_embedding, limit=settings.memory_recall_vector_top_k, include_inactive=include_pending_review)
        fallback = self.store.list_memories(user_id, include_inactive=include_pending_review)[:settings.memory_recall_fallback_limit]
        by_id = {m.id: m for m in fallback}; by_id.update({m.id: m for m in vector_candidates})
        for memory in [m for m in by_id.values() if m.status in allowed]:
            if memory.embedding is None: memory.embedding = self._embed_text(memory.content)
            score, signals = hybrid_retrieval_score(query, memory, query_tags=query_tags, query_embedding=query_embedding)
            if score <= 0: continue
            self.store.update_memory(memory.id, actor="recall", last_recalled_at=utc_now())
            results.append(RecallResult(memory=memory, score=score, explanation=RecallExplanation(memory.source_session, memory.confidence_score, memory.updated_at, signals, self._reasoning_path(query, memory, signals))))
        if hasattr(self.store, "list_memory_stream"):
            accessed_stream_ids: list[str] = []
            for entry in self.store.list_memory_stream(user_id, include_inactive=include_pending_review)[-settings.memory_recall_fallback_limit:]:
                if entry.status not in allowed:
                    continue
                memory = memory_from_stream_entry(entry)
                score, signals = hybrid_retrieval_score(query, memory, query_tags=query_tags, query_embedding=query_embedding)
                if score <= 0:
                    continue
                accessed_stream_ids.append(entry.id)
                results.append(RecallResult(memory=memory, score=score, explanation=RecallExplanation(memory.source_session, memory.confidence_score, memory.updated_at, signals, self._reasoning_path(query, memory, signals))))
            if accessed_stream_ids and hasattr(self.store, "update_memory_stream_access"):
                self.store.update_memory_stream_access(accessed_stream_ids, actor="recall")
        return conflict_aware_recall_filter(results, limit)

    def forget(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        memory = self._owned_memory(user_id, memory_id)
        updated = self.store.update_memory(memory.id, actor=actor, status=MemoryStatus.FORGOTTEN)
        self.sync_memory_vector(updated); return updated
    def approve(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        updated = self.store.update_memory(self._owned_memory(user_id, memory_id).id, actor=actor, status=MemoryStatus.ACTIVE, approved_at=utc_now(), last_confirmed_at=utc_now())
        self.sync_memory_vector(updated); return updated
    def reject(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        updated = self.store.update_memory(self._owned_memory(user_id, memory_id).id, actor=actor, status=MemoryStatus.REJECTED)
        self.sync_memory_vector(updated); return updated
    def archive(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        updated = self.store.update_memory(self._owned_memory(user_id, memory_id).id, actor=actor, status=MemoryStatus.ARCHIVED)
        self.sync_memory_vector(updated); return updated
    def edit_and_approve(self, user_id: str, memory_id: str, *, content: str, actor: str = "user") -> Memory:
        updated = self.store.update_memory(self._owned_memory(user_id, memory_id).id, actor=actor, content=content, embedding=self._embed_text(content), status=MemoryStatus.ACTIVE, approved_at=utc_now(), last_confirmed_at=utc_now())
        self.sync_memory_vector(updated); return updated
    def inspect(self, user_id: str, *, include_inactive: bool = False) -> list[Memory]: return self.store.list_memories(user_id, include_inactive=include_inactive)
    def maintenance(self, user_id: str) -> dict[str, int]:
        memories = self.store.list_memories(user_id, include_inactive=True)
        merged = self._merge_duplicates(memories)
        promoted = 0
        for memory in self.store.list_memories(user_id):
            if memory.memory_type != MemoryType.SEMANTIC and memory.stability_score >= 0.7 and memory.confidence_score >= 0.75:
                self.store.update_memory(memory.id, actor="profile-agent", memory_type=MemoryType.SEMANTIC)
                promoted += 1
        return {"merged": merged, "promoted": promoted, "decayed": 0, "archived": 0}

    def sync_memory_vector(self, memory: Memory) -> None:
        if should_index_memory(memory): self.upsert_active_memory_vector(memory)
        else: self.delete_memory_vector(memory.id)
    def upsert_active_memory_vector(self, memory: Memory) -> None:
        if not self.vector_index: return
        if not should_index_memory(memory): self.delete_memory_vector(memory.id); return
        self.vector_index.upsert_memory(memory)
    def delete_memory_vector(self, memory_id: str) -> None:
        if self.vector_index and hasattr(self.vector_index, "delete_memory"): self.vector_index.delete_memory(memory_id)
    def reconcile_vectors(self) -> dict[str, list[str] | int]:
        active = {m.id: m for uid in self.store.list_user_ids() for m in self.store.list_memories(uid)}
        indexed = set(self.vector_index.list_memory_ids()) if self.vector_index and hasattr(self.vector_index, "list_memory_ids") else set()
        missing = sorted(set(active) - indexed); stale = sorted(indexed - set(active))
        for mid in stale: self.delete_memory_vector(mid)
        for mid in missing: self.upsert_active_memory_vector(active[mid])
        return {"missing_active_vectors": missing, "orphan_or_stale_vectors": stale, "fixed": len(missing)+len(stale)}

    def _owned_memory(self, user_id: str, memory_id: str) -> Memory:
        m = self.store.get_memory(memory_id)
        if m.user_id != user_id: raise ValueError("memory does not belong to user")
        return m
    def _embed_text(self, text: str) -> list[float]:
        if self.embedding_provider is None: return [0.0] * self.fallback_embedding_dimensions
        return self.embedding_provider.embed_text(text)
    def _reasoning_path(self, query: str, memory: Memory, signals: dict[str, float]) -> list[str]:
        return [f"Matched query tokens: {sorted(tokenize(query) & tokenize(memory.content))}", "Top ranking signals: " + ", ".join(f"{k}={v:.2f}" for k,v in sorted(signals.items(), key=lambda i:i[1], reverse=True)[:3]), f"Source session: {memory.source_session}"]
    def _merge_duplicates(self, memories: list[Memory]) -> int:
        merged=0; by_sig=defaultdict(list)
        for m in memories: by_sig[(m.memory_type.value, m.metadata.get("key"), str(m.metadata.get("value", "")).lower() or tuple(sorted(tokenize(m.content))))].append(m)
        for dups in by_sig.values():
            if len(dups)<2: continue
            keeper=max(dups,key=lambda m:m.confidence_score)
            for dup in dups:
                if dup.id!=keeper.id:
                    self.store.update_memory(dup.id, actor="compaction-agent", status=MemoryStatus.ARCHIVED); self.sync_memory_vector(dup); merged+=1
        return merged


def conflict_aware_recall_filter(results: list[RecallResult], limit: int) -> list[RecallResult]:
    """Keep only current, confirmed memories and one winner per canonical key/topic."""

    allowed = {MemoryStatus.ACTIVE}
    ranked = sorted((item for item in results if item.memory.status in allowed), key=lambda item: item.score, reverse=True)
    by_key: dict[str, RecallResult] = {}
    unkeyed: list[RecallResult] = []
    for item in ranked:
        key = str(item.memory.metadata.get("key") or "")
        if not key:
            unkeyed.append(item)
            continue
        current = by_key.get(key)
        if current is None or (item.memory.updated_at, item.score) > (current.memory.updated_at, current.score):
            by_key[key] = item
    preferred = sorted(by_key.values(), key=lambda item: (item.memory.memory_type == MemoryType.CONVERSATION_SUMMARY, item.score), reverse=True)
    return (preferred + unkeyed)[:limit]

def normalize_candidate(candidate: MemoryCandidate) -> tuple[str, str | None, str]:
    key = candidate.key or infer_key(candidate.content)
    value = str(candidate.value or infer_value(candidate.content, key) or "").strip()
    content = candidate.content or (f"User's {key} is {value}." if key and value else candidate.content)
    return content, key, value

def infer_key(content: str) -> str | None:
    lower=content.lower()
    if re.search(r"\bmy name is\b|user'?s name is", lower): return "profile.name"
    if "email" in lower: return "profile.email"
    if "company" in lower or "work at" in lower: return "profile.occupation"
    if "years old" in lower or re.search(r"\bage\b", lower): return "profile.age"
    if "timezone" in lower: return "profile.timezone"
    if "language" in lower and ("prefer" in lower or "speaks" in lower): return "preference.language"
    if "theme" in lower: return "preference.ui_theme"
    return _subject_key(content)

def infer_value(content: str, key: str | None) -> str | None:
    if key == "profile.name":
        m=re.search(r"(?:my name is|user'?s name is|i am|i'm)\s+([A-Z][\w'-]*)", content, re.I); return m.group(1) if m else None
    if key == "profile.age":
        m=re.search(r"\b(\d{1,3})\s*(?:years old|yo|y/o)?\b", content, re.I); return m.group(1) if m else None
    return None

def values_equal(a: str, b: str) -> bool: return a.strip().casefold() == b.strip().casefold()
def conflict_type_for_key(key: str | None) -> str:
    if key and key.startswith("profile."): return "identity_conflict"
    if key and key.startswith("preference."): return "preference_conflict"
    return "fact_conflict"
def confirmation_prompt(existing: Memory, candidate: Memory) -> str:
    return f"I currently remember {friendly_memory(existing)}. Should I update it to {candidate.metadata.get('value') or candidate.content}?"
def friendly_memory(memory: Memory) -> str:
    if memory.metadata.get("key") == "profile.name" and memory.metadata.get("value"): return f"your name as {memory.metadata['value']}"
    return memory.content

def classify_conflict_reply(reply: str, detector: ConflictDetector | None = None) -> str:
    text=reply.strip().lower()
    if re.search(r"^(yes|yep|yeah|correct|update|sure)\b", text) or "update it" in text: return "accept"
    if re.search(r"^(no|nope|keep|wrong|do not|don't)\b", text) or "keep" in text: return "reject"
    if detector is not None:
        try:
            raw=detector.chat([{"role":"system","content":"Return JSON only: {\"decision\":\"accept|reject|merge\"}"},{"role":"user","content":reply}], model=settings.qwen_flash_model, temperature=0, max_tokens=settings.qwen_conflict_resolution_max_tokens)
            return {"accept":"accept","reject":"reject"}.get(json.loads(raw).get("decision"), "merge")
        except Exception: pass
    return "merge"

def _subject_key(content: str) -> str | None:
    tokens = TOKEN_RE.findall(content.lower())
    for index, token in enumerate(tokens):
        if token in {"uses", "prefers", "likes", "works", "runs"} and index > 0: return tokens[index - 1] + ":" + token
    return None


def policy_status_for_candidate(candidate: MemoryCandidate, key: str | None) -> MemoryStatus:
    """Decide auto-store policy from type, importance, durability, conflict, and sensitivity."""
    memory_type = MemoryType(candidate.type)
    durable = key is not None or memory_type in {MemoryType.USER_FACT, MemoryType.PREFERENCE, MemoryType.PROJECT_CONTEXT, MemoryType.WORKFLOW}
    important = stream_importance_1_to_10(candidate.content, memory_type, candidate.confidence) >= 4
    low_risk_sensitive = candidate.sensitivity in {"low", "medium"} and (bool(key) or memory_type in {MemoryType.USER_FACT, MemoryType.PREFERENCE} or "name" in candidate.content.lower())
    auto_store = candidate.source == "user_message" and candidate.confidence >= 0.70 and durable and important and low_risk_sensitive
    if candidate.sensitivity == "high":
        auto_store = False
    return MemoryStatus.ACTIVE if auto_store else MemoryStatus.PENDING_REVIEW
