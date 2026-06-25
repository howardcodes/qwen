"""Core memory operating system implementation.

- MemoryOS manages memories for an AI agent:
    - remember(): create a new memory with quality scoring and conflict resolution
    - recall(): retrieve relevant memories with explainable scoring signals
    - forget(): mark a memory as forgotten while retaining audit history
    - inspect(): list a user's memories for transparency controls
    - maintenance(): run autonomous memory maintenance to merge duplicates, promote stable facts, decay low-stability memories, and archive stale memories


"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import json
from typing import Protocol

from .models import (
    Memory,
    MemoryEdge,
    MemoryStatus,
    MemoryType,
    RecallExplanation,
    RecallResult,
    RelationType,
    clamp_score,
    utc_now,
)
from .scoring import TOKEN_RE, hybrid_retrieval_score, keyword_score, quality_scores, tokenize
from .store import InMemoryStore


class ConflictDetector(Protocol):
    """Provider interface for Qwen semantic conflict checks."""

    def chat(self, messages: Sequence[object], *, model: str | None = None, temperature: float = 0.2) -> str:
        """Return a JSON conflict decision."""


class EmbeddingProvider(Protocol):
    """Provider interface for QwenCloud/Alibaba text embeddings."""

    def embed_text(self, text: str) -> list[float]:
        """Embed one text value."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text values preserving input order."""


class MemoryOS:
    """Self-correcting memory manager for AI agents."""

    def __init__(
        self,
        store: InMemoryStore | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        require_live_embeddings: bool = False,
        fallback_embedding_dimensions: int = 1024,
        conflict_detector: ConflictDetector | None = None,
    ) -> None:
        self.store = store or InMemoryStore()
        self.embedding_provider = embedding_provider
        self.require_live_embeddings = require_live_embeddings
        self.fallback_embedding_dimensions = fallback_embedding_dimensions
        self.conflict_detector = conflict_detector

    def remember(
        self,
        *,
        user_id: str,
        content: str,
        memory_type: MemoryType | str = MemoryType.EPISODIC,
        source_session: str,
        tags: Iterable[str] = (),
        confidence_score: float | None = None,
        importance_score: float | None = None,
        novelty_score: float | None = None,
        stability_score: float | None = None,
        metadata: dict[str, object] | None = None,
        confidence_reasons: list[str] | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        sensitivity: str = "low",
        actor: str = "memory-agent",
    ) -> Memory:
        """Create a memory with duplicate and conflict lifecycle controls."""

        existing = self.store.list_memories(user_id)
        scores = quality_scores(content, existing) # KIV In production, this would be a more sophisticated scoring function possibly involving ML models
        memory = Memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            source_session=source_session,
            confidence_score=confidence_score if confidence_score is not None else float(scores["confidence"]),
            importance_score=importance_score if importance_score is not None else float(scores["importance"]),
            novelty_score=novelty_score if novelty_score is not None else float(scores["novelty"]),
            stability_score=stability_score if stability_score is not None else float(scores["stability"]),
            tags=set(tags),
            metadata=dict(metadata or {}),
            confidence_reasons=confidence_reasons or list(scores.get("confidence_reasons", [])),
            embedding=self._embed_text(content),
            status=status,
            sensitivity=sensitivity,
            approved_at=utc_now() if status == MemoryStatus.ACTIVE else None,
            last_seen_at=utc_now(),
        )
        duplicate = self._handle_duplicate(memory, actor=actor)
        if duplicate is not None:
            return duplicate
        self._mark_conflicts(memory, actor=actor)
        self.store.add_memory(memory, actor=actor)
        return memory

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        query_tags: Iterable[str] = (),
        limit: int = 5,
        include_pending_review: bool = False,
    ) -> list[RecallResult]:
        """Return ranked memories with explainable recall metadata.

        Normal recall returns only approved active memories. Agent chat can opt in
        to pending-review facts so it can transparently tell the user that a
        matching memory exists but still needs approval instead of saying the
        memory does not exist.
        """

        results: list[RecallResult] = []
        query_embedding = self._embed_text(query)
        allowed_statuses = {MemoryStatus.ACTIVE}
        if include_pending_review:
            allowed_statuses.add(MemoryStatus.PENDING_REVIEW)
        vector_candidates = self.store.vector_search(
            user_id,
            query_embedding,
            limit=max(limit * 4, 20),
            include_inactive=include_pending_review,
        )
        fallback_candidates = self.store.list_memories(user_id, include_inactive=include_pending_review)
        candidate_by_id = {memory.id: memory for memory in fallback_candidates}
        candidate_by_id.update({memory.id: memory for memory in vector_candidates})
        candidate_memories = [memory for memory in candidate_by_id.values() if memory.status in allowed_statuses]
        for memory in candidate_memories:
            if memory.embedding is None:
                memory.embedding = self._embed_text(memory.content)
            score, signals = hybrid_retrieval_score(
                query,
                memory,
                query_tags=query_tags,
                query_embedding=query_embedding,
            )
            if score <= 0:
                continue
            memory.last_recalled_at = utc_now()
            results.append(
                RecallResult(
                    memory=memory,
                    score=score,
                    explanation=RecallExplanation(
                        source_session=memory.source_session,
                        confidence_score=memory.confidence_score,
                        timestamp=memory.updated_at,
                        ranking_signals=signals,
                        reasoning_path=self._reasoning_path(query, memory, signals),
                    ),
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit] # most relevant memories at the top

    def forget(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        """Mark a memory as forgotten while retaining audit history."""

        memory = self.store.get_memory(memory_id)
        if memory.user_id != user_id:
            raise ValueError("memory does not belong to user")
        return self.store.update_memory(memory_id, actor=actor, status=MemoryStatus.FORGOTTEN)

    def approve(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        memory = self._owned_memory(user_id, memory_id)
        return self.store.update_memory(memory.id, actor=actor, status=MemoryStatus.ACTIVE, approved_at=utc_now())

    def reject(self, user_id: str, memory_id: str, *, actor: str = "user") -> Memory:
        memory = self._owned_memory(user_id, memory_id)
        return self.store.update_memory(memory.id, actor=actor, status=MemoryStatus.REJECTED)

    def edit_and_approve(self, user_id: str, memory_id: str, *, content: str, actor: str = "user") -> Memory:
        memory = self._owned_memory(user_id, memory_id)
        return self.store.update_memory(memory.id, actor=actor, content=content, embedding=self._embed_text(content), status=MemoryStatus.ACTIVE, approved_at=utc_now())

    def _owned_memory(self, user_id: str, memory_id: str) -> Memory:
        memory = self.store.get_memory(memory_id)
        if memory.user_id != user_id:
            raise ValueError("memory does not belong to user")
        return memory

    def inspect(self, user_id: str, *, include_inactive: bool = False) -> list[Memory]:
        """List a user's memories for transparency controls."""

        return self.store.list_memories(user_id, include_inactive=include_inactive)

    def maintenance(self, user_id: str) -> dict[str, int]:
        """Run autonomous memory maintenance for one user."""

        memories = self.store.list_memories(user_id, include_inactive=True)
        merged = self._merge_duplicates([m for m in memories if m.status != MemoryStatus.FORGOTTEN])
        promoted = self._promote_stable_facts(self.store.list_memories(user_id))
        decayed = self._decay_low_stability(self.store.list_memories(user_id))
        archived = self._archive_stale(self.store.list_memories(user_id))
        return {"merged": merged, "promoted": promoted, "decayed": decayed, "archived": archived}

    def _embed_text(self, text: str) -> list[float]:
        """Embed text through the configured Qwen/Alibaba embedding provider."""

        if self.embedding_provider is None:
            raise RuntimeError("A Qwen embedding provider is required; local embeddings are disabled")
        try:
            return self.embedding_provider.embed_text(text)
        except Exception:
            if self.require_live_embeddings:
                raise
            raise RuntimeError("Qwen embedding provider failed and local embeddings are disabled")


    def _handle_duplicate(self, memory: Memory, *, actor: str) -> Memory | None:
        candidates = self.store.vector_search(memory.user_id, memory.embedding or [], limit=5, include_inactive=True)
        for candidate in candidates:
            lexical_similarity = keyword_score(memory.content, candidate)
            similarity = lexical_similarity
            if memory.embedding and candidate.embedding:
                from .scoring import cosine_similarity
                similarity = max(similarity, cosine_similarity(memory.embedding, candidate.embedding))
            if lexical_similarity > 0.90:
                try:
                    from .monitoring import memory_metrics as metrics
                    metrics.duplicate_count.inc()
                except Exception:
                    pass
                memory.status = MemoryStatus.ARCHIVED
                memory.tags.add("duplicate")
                memory.metadata["duplicate_of"] = candidate.id
                memory.confidence_reasons.append("duplicate archived for audit and maintenance")
                return None
            if lexical_similarity > 0.75:
                memory.status = MemoryStatus.PENDING_REVIEW
                memory.tags.add("possible_duplicate")
                memory.metadata["possible_duplicate_of"] = candidate.id
                memory.confidence_reasons.append("possible duplicate requires review")
                break
        return None

    def _mark_conflicts(self, memory: Memory, *, actor: str) -> None:
        for candidate in self.store.vector_search(memory.user_id, memory.embedding or [], limit=5, include_inactive=True):
            if candidate.status == MemoryStatus.FORGOTTEN or candidate.id == memory.id:
                continue
            conflict, reason = self._semantic_conflict(memory, candidate)
            if conflict:
                try:
                    from .monitoring import memory_metrics as metrics
                    metrics.conflict_count.inc()
                except Exception:
                    pass
                self.store.add_edge(MemoryEdge(source_memory=memory.id, target_memory=candidate.id, relation_type=RelationType.CONTRADICTS))
                if memory.confidence_score >= candidate.confidence_score:
                    self.store.update_memory(candidate.id, actor="conflict-resolver", status=MemoryStatus.DEPRECATED)
                    self.store.add_edge(MemoryEdge(source_memory=memory.id, target_memory=candidate.id, relation_type=RelationType.SUPERSEDES))
                    return
                memory.status = MemoryStatus.POSSIBLY_CONFLICTING
                memory.conflicting_memory_id = candidate.id
                memory.conflict_reason = reason
                memory.confidence_reasons.append(f"conflict requires review: {reason}")
                return

    def _semantic_conflict(self, memory: Memory, candidate: Memory) -> tuple[bool, str]:
        if self.conflict_detector is not None:
            prompt = f"Are these memories semantically conflicting? Return JSON with conflict and reason.\nA: {memory.content}\nB: {candidate.content}"
            try:
                data = json.loads(self.conflict_detector.chat([{"role": "user", "content": prompt}], temperature=0))
                return bool(data.get("conflict")), str(data.get("reason", "semantic conflict"))
            except Exception:
                pass
        new_subject = _subject_key(memory.content)
        if new_subject and new_subject == _subject_key(candidate.content) and keyword_score(memory.content, candidate) < 0.85:
            return True, "same subject has a different asserted value"
        return False, ""

    def _resolve_conflicts(self, new_memory: Memory, existing: list[Memory]) -> None:
        new_subject = _subject_key(new_memory.content)
        if not new_subject:
            return
        for old_memory in existing:
            if _subject_key(old_memory.content) != new_subject:
                continue
            if keyword_score(new_memory.content, old_memory) > 0.85:
                continue

            # relationships between memories
            if new_memory.confidence_score >= old_memory.confidence_score and new_memory.metadata.get("source", "explicit_user") == "explicit_user":
                self.store.update_memory(
                    old_memory.id,
                    actor="conflict-resolver",
                    status=MemoryStatus.POSSIBLY_CONFLICTING,
                )
                self.store.add_edge(
                    MemoryEdge(
                        source_memory=new_memory.id,
                        target_memory=old_memory.id,
                        relation_type=RelationType.SUPERSEDES,
                    )
                )
            self.store.add_edge(
                MemoryEdge(
                    source_memory=new_memory.id,
                    target_memory=old_memory.id,
                    relation_type=RelationType.CONTRADICTS,
                )
            )
            if new_memory.metadata.get("source", "explicit_user") != "explicit_user":
                self.store.update_memory(
                    new_memory.id,
                    actor="conflict-resolver",
                    status=MemoryStatus.POSSIBLY_CONFLICTING,
                    confidence_reasons=[*new_memory.confidence_reasons, "possible contradiction requires review"],
                )

    def _reasoning_path(
        self,
        query: str,
        memory: Memory,
        signals: dict[str, float],
    ) -> list[str]:
        path = [f"Matched query tokens: {sorted(tokenize(query) & tokenize(memory.content))}"]
        strongest = sorted(signals.items(), key=lambda item: item[1], reverse=True)[:3]
        path.append("Top ranking signals: " + ", ".join(f"{name}={value:.2f}" for name, value in strongest))
        if memory.tags:
            path.append("Memory tags: " + ", ".join(sorted(memory.tags)))
        path.append(f"Source session: {memory.source_session}")
        return path

    def _merge_duplicates(self, memories: list[Memory]) -> int:
        merged = 0
        by_signature: dict[tuple[str, ...], list[Memory]] = defaultdict(list)
        for memory in memories:
            by_signature[tuple(sorted(tokenize(memory.content)))].append(memory)
        # This is a simple heuristic for finding duplicates based on token overlap. In production, this could be more sophisticated and involve ML models to identify paraphrased duplicates or related facts.
        for duplicates in by_signature.values():
            if len(duplicates) < 2:
                continue
            keeper = max(duplicates, key=lambda item: item.confidence_score + item.importance_score)
            for duplicate in duplicates:
                if duplicate.id == keeper.id:
                    continue
                self.store.update_memory(
                    duplicate.id,
                    actor="compaction-agent",
                    status=MemoryStatus.ARCHIVED,
                )
                self.store.add_edge(
                    MemoryEdge(
                        source_memory=keeper.id,
                        target_memory=duplicate.id,
                        relation_type=RelationType.SUPERSEDES,
                    )
                )
                merged += 1
        return merged

    def _promote_stable_facts(self, memories: list[Memory]) -> int:
        promoted = 0
        for memory in memories:
            if memory.memory_type == MemoryType.SEMANTIC:
                continue
            if memory.stability_score >= 0.7 and memory.confidence_score >= 0.75:
                self.store.update_memory(
                    memory.id,
                    actor="profile-agent",
                    memory_type=MemoryType.SEMANTIC,
                    importance_score=clamp_score(memory.importance_score + 0.1),
                )
                promoted += 1
        return promoted

    def _decay_low_stability(self, memories: list[Memory]) -> int:
        decayed = 0
        for memory in memories:
            if memory.stability_score < 0.4:
                self.store.update_memory(
                    memory.id,
                    actor="maintenance-agent",
                    confidence_score=clamp_score(memory.confidence_score - 0.05),
                )
                decayed += 1
        return decayed

    def _archive_stale(self, memories: list[Memory]) -> int:
        archived = 0
        for memory in memories:
            if memory.confidence_score <= 0.2 and memory.importance_score <= 0.3:
                self.store.update_memory(
                    memory.id,
                    actor="maintenance-agent",
                    status=MemoryStatus.ARCHIVED,
                )
                archived += 1
        return archived


def _subject_key(content: str) -> str | None:
    tokens = TOKEN_RE.findall(content.lower())
    if not tokens:
        return None
    subject_markers = {"uses", "prefers", "likes", "works", "runs"}
    for index, token in enumerate(tokens):
        if token in subject_markers and index > 0:
            return tokens[index - 1] + ":" + token
    return None
