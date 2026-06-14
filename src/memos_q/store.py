"""Storage adapters for MemOS-Q."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import AuditEvent, Memory, MemoryEdge, MemoryStatus, RelationType, utc_now


class InMemoryStore:
    """Simple repository used by the prototype and tests."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, MemoryEdge] = {}
        self._audit_log: list[AuditEvent] = []
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
        "status": memory.status.value,
        "version": memory.version,
        "tags": sorted(memory.tags),
    }
