"""Storage adapters for MemOS-Q."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import AuditEvent, Memory, MemoryConflict, MemoryConflictResolution, MemoryConflictStatus, MemoryEdge, MemoryStatus, RelationType, utc_now
from .scoring import cosine_similarity


class InMemoryStore:
    """Simple repository used by the prototype and tests."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, MemoryEdge] = {}
        self._audit_log: list[AuditEvent] = []
        self._conflicts: dict[str, MemoryConflict] = {}
        self._by_user: dict[str, set[str]] = defaultdict(set)

    def add_memory(self, memory: Memory, *, actor: str = "memory-agent") -> Memory:
        self._memories[memory.id] = memory
        self._by_user[memory.user_id].add(memory.id)
        self.record_audit("remember", actor, memory.id, None, memory_snapshot(memory))
        return memory

    def update_memory(
        self,
        memory_id: str,
        *,
        actor: str = "memory-agent",
        **changes: object,
    ) -> Memory:
        memory = self.get_memory(memory_id)
        previous = memory_snapshot(memory)
        for key, value in changes.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        memory.version += 1
        memory.updated_at = utc_now()
        self.record_audit("update", actor, memory.id, previous, memory_snapshot(memory))
        return memory

    def get_memory(self, memory_id: str) -> Memory:
        return self._memories[memory_id]

    def list_memories(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[Memory]:
        memories = [self._memories[memory_id] for memory_id in self._by_user[user_id]]
        if include_inactive:
            return sorted(memories, key=lambda item: item.created_at)
        return sorted(
            [memory for memory in memories if memory.status == MemoryStatus.ACTIVE],
            key=lambda item: item.created_at,
        )

    def vector_search(self, user_id: str, query_embedding: list[float], *, limit: int = 20, include_inactive: bool = False) -> list[Memory]:
        """Return nearest memories for a user using stored embedding vectors."""

        candidates = [memory for memory in self.list_memories(user_id, include_inactive=include_inactive) if memory.embedding and memory.status != MemoryStatus.FORGOTTEN]
        return sorted(
            candidates,
            key=lambda memory: cosine_similarity(query_embedding, memory.embedding),
            reverse=True,
        )[:limit]

    def add_conflict(self, conflict: MemoryConflict) -> MemoryConflict:
        self._conflicts[conflict.id] = conflict
        self.record_audit("conflict_create", "memory-conflict", conflict.candidate_memory_id, None, conflict_snapshot(conflict))
        return conflict

    def get_conflict(self, conflict_id: str) -> MemoryConflict:
        return self._conflicts[conflict_id]

    def pending_conflict_for_user(self, user_id: str) -> MemoryConflict | None:
        pending = [c for c in self._conflicts.values() if c.user_id == user_id and c.status == MemoryConflictStatus.PENDING]
        return sorted(pending, key=lambda c: c.created_at)[-1] if pending else None

    def resolve_conflict(self, conflict_id: str, *, resolution: str, actor: str = "memory-agent") -> MemoryConflict:
        conflict = self.get_conflict(conflict_id)
        previous = conflict_snapshot(conflict)
        conflict.status = MemoryConflictStatus.RESOLVED
        conflict.resolution = resolution
        conflict.resolved_at = utc_now()
        self.record_audit("conflict_resolve", actor, conflict.candidate_memory_id, previous, conflict_snapshot(conflict))
        return conflict

    def list_conflicts(self, user_id: str | None = None, *, include_resolved: bool = False) -> list[MemoryConflict]:
        conflicts = list(self._conflicts.values())
        if user_id is not None:
            conflicts = [c for c in conflicts if c.user_id == user_id]
        if not include_resolved:
            conflicts = [c for c in conflicts if c.status == MemoryConflictStatus.PENDING]
        return sorted(conflicts, key=lambda c: c.created_at)

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        self._edges[edge.id] = edge
        self.record_audit(
            "edge_create",
            "memory-graph",
            edge.source_memory,
            None,
            {
                "source_memory": edge.source_memory,
                "target_memory": edge.target_memory,
                "relation_type": edge.relation_type.value,
            },
        )
        return edge

    def edges_for(self, memory_id: str, relation_type: RelationType | None = None) -> list[MemoryEdge]:
        edges = [
            edge
            for edge in self._edges.values()
            if edge.source_memory == memory_id or edge.target_memory == memory_id
        ]
        if relation_type is None:
            return edges
        return [edge for edge in edges if edge.relation_type == relation_type]

    def list_user_ids(self) -> list[str]:
        """Return user ids with at least one memory."""

        return sorted(self._by_user)

    def audit_log(self, memory_id: str | None = None) -> list[AuditEvent]:
        if memory_id is None:
            return list(self._audit_log)
        return [event for event in self._audit_log if event.memory_id == memory_id]

    def record_audit(
        self,
        action: str,
        actor: str,
        memory_id: str,
        previous_value: dict[str, object] | None,
        new_value: dict[str, object] | None,
    ) -> None:
        self._audit_log.append(
            AuditEvent(
                action=action,
                actor=actor,
                memory_id=memory_id,
                previous_value=previous_value,
                new_value=new_value,
            )
        )

    def bulk_update(self, memories: Iterable[Memory], *, actor: str, action: str) -> None:
        for memory in memories:
            previous = memory_snapshot(memory)
            memory.version += 1
            memory.updated_at = utc_now()
            self.record_audit(action, actor, memory.id, previous, memory_snapshot(memory))


def memory_snapshot(memory: Memory) -> dict[str, object]:
    """Return a JSON-like representation for audit events."""

    return {
        "id": memory.id,
        "user_id": memory.user_id,
        "content": memory.content,
        "memory_type": memory.memory_type.value,
        "source_session": memory.source_session,
        "confidence_score": memory.confidence_score,
        "importance_score": memory.importance_score,
        "novelty_score": memory.novelty_score,
        "stability_score": memory.stability_score,
        "confidence_reasons": list(memory.confidence_reasons),
        "embedding_dimensions": len(memory.embedding or []),
        "sensitivity": memory.sensitivity,
        "approved_at": memory.approved_at.isoformat() if memory.approved_at else None,
        "last_confirmed_at": memory.last_confirmed_at.isoformat() if memory.last_confirmed_at else None,
        "key": memory.metadata.get("key"),
        "value": memory.metadata.get("value"),
        "last_seen_at": memory.last_seen_at.isoformat() if memory.last_seen_at else None,
        "conflicting_memory_id": memory.conflicting_memory_id,
        "conflict_reason": memory.conflict_reason,
        "status": memory.status.value,
        "version": memory.version,
        "tags": sorted(memory.tags),
    }


def conflict_snapshot(conflict: MemoryConflict) -> dict[str, object]:
    return {
        "id": conflict.id,
        "user_id": conflict.user_id,
        "existing_memory_id": conflict.existing_memory_id,
        "candidate_memory_id": conflict.candidate_memory_id,
        "conflict_type": conflict.conflict_type,
        "existing_content": conflict.existing_content,
        "candidate_content": conflict.candidate_content,
        "status": conflict.status.value,
        "created_at": conflict.created_at.isoformat(),
        "resolved_at": conflict.resolved_at.isoformat() if conflict.resolved_at else None,
        "resolution": conflict.resolution,
    }
