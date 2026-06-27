"""Storage adapters for MemOS-Q."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any, TypeVar

from .models import AuditEvent, ChatTurn, Memory, UserProfile, MemoryStreamEntry, MemoryConflict, MemoryConflictStatus, MemoryEdge, MemoryStatus, RelationType, SessionState, utc_now
from .scoring import cosine_similarity

T = TypeVar("T")


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _dataclass_payload(instance: Any) -> dict[str, Any]:
    return {field.name: _json_ready(getattr(instance, field.name)) for field in fields(instance)}


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _load_model(cls: type[T], payload: dict[str, Any]) -> T:
    data = dict(payload)
    for field in fields(cls):
        if field.name not in data:
            continue
        if field.type is datetime or "datetime" in str(field.type):
            data[field.name] = _parse_datetime(data[field.name])
    if cls is Memory and isinstance(data.get("tags"), list):
        data["tags"] = set(data["tags"])
    return cls(**data)


class InMemoryStore:
    """Simple repository used by the prototype and tests."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._edges: dict[str, MemoryEdge] = {}
        self._audit_log: list[AuditEvent] = []
        self._conflicts: dict[str, MemoryConflict] = {}
        self._stream: dict[str, MemoryStreamEntry] = {}
        self._stream_by_user: dict[str, set[str]] = defaultdict(set)
        self._by_user: dict[str, set[str]] = defaultdict(set)
        self._profiles: dict[str, UserProfile] = {}
        self._conversation_turns: dict[tuple[str, str], list[ChatTurn]] = defaultdict(list)
        self._session_state: dict[tuple[str, str], SessionState] = {}


    def get_user_profile(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    def append_conversation_turn(self, user_id: str, conversation_id: str, turn: ChatTurn, *, max_turns: int = 50) -> None:
        key = (user_id, conversation_id)
        self._conversation_turns[key].append(turn)
        self._conversation_turns[key] = self._conversation_turns[key][-max_turns:]
        self.record_audit("conversation_turn_append", "conversation_turns", conversation_id, None, {"role": turn.role, "content": turn.content})

    def recent_conversation_turns(self, user_id: str, conversation_id: str, *, limit: int = 10) -> list[ChatTurn]:
        return self._conversation_turns[(user_id, conversation_id)][-limit:]

    def get_session_state(self, user_id: str, conversation_id: str) -> SessionState:
        return self._session_state.get((user_id, conversation_id), SessionState())

    def update_session_state(self, user_id: str, conversation_id: str, state: SessionState) -> SessionState:
        state.updated_at = utc_now()
        self._session_state[(user_id, conversation_id)] = state
        self.record_audit("session_state_update", "session_state", conversation_id, None, session_state_snapshot(state))
        return state

    def upsert_user_profile(self, user_id: str, *, actor: str = "profile-agent", **changes: object) -> UserProfile:
        profile = self._profiles.get(user_id) or UserProfile(user_id=user_id)
        previous = profile_snapshot(profile) if user_id in self._profiles else None
        for key, value in changes.items():
            if hasattr(profile, key) and value not in (None, ""):
                setattr(profile, key, value)
        profile.updated_at = utc_now()
        self._profiles[user_id] = profile
        self.record_audit("profile_upsert", actor, user_id, previous, profile_snapshot(profile))
        return profile

    def add_memory_stream_entry(self, entry: MemoryStreamEntry, *, actor: str = "memory-stream") -> MemoryStreamEntry:
        self._stream[entry.id] = entry
        self._stream_by_user[entry.user_id].add(entry.id)
        self.record_audit("stream_append", actor, entry.id, None, memory_stream_snapshot(entry))
        return entry

    def list_memory_stream(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryStreamEntry]:
        entries = [self._stream[entry_id] for entry_id in self._stream_by_user[user_id]]
        if not include_inactive:
            entries = [entry for entry in entries if entry.status == MemoryStatus.ACTIVE]
        return sorted(entries, key=lambda item: item.created_at)

    def update_memory_stream_access(self, entry_ids: Iterable[str], *, actor: str = "recall") -> None:
        for entry_id in entry_ids:
            if entry_id in self._stream:
                previous = memory_stream_snapshot(self._stream[entry_id])
                self._stream[entry_id].last_accessed_at = utc_now()
                self.record_audit("stream_access", actor, entry_id, previous, memory_stream_snapshot(self._stream[entry_id]))

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


class JsonFileMemoryStore(InMemoryStore):
    """Durable local store that persists the in-memory repository to JSON."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._loading = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        self._loading = True
        try:
            payload = json.loads(self.path.read_text())
            for item in payload.get("memories", []):
                memory = _load_model(Memory, item)
                self._memories[memory.id] = memory
                self._by_user[memory.user_id].add(memory.id)
            for item in payload.get("stream", []):
                entry = _load_model(MemoryStreamEntry, item)
                self._stream[entry.id] = entry
                self._stream_by_user[entry.user_id].add(entry.id)
            for item in payload.get("profiles", []):
                profile = _load_model(UserProfile, item)
                self._profiles[profile.user_id] = profile
            for item in payload.get("conflicts", []):
                conflict = _load_model(MemoryConflict, item)
                self._conflicts[conflict.id] = conflict
            for item in payload.get("edges", []):
                edge = _load_model(MemoryEdge, item)
                self._edges[edge.id] = edge
            self._audit_log = [_load_model(AuditEvent, item) for item in payload.get("audit_log", [])]
        finally:
            self._loading = False

    def _persist(self) -> None:
        if self._loading:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memories": [_dataclass_payload(item) for item in self._memories.values()],
            "stream": [_dataclass_payload(item) for item in self._stream.values()],
            "profiles": [_dataclass_payload(item) for item in self._profiles.values()],
            "conflicts": [_dataclass_payload(item) for item in self._conflicts.values()],
            "edges": [_dataclass_payload(item) for item in self._edges.values()],
            "audit_log": [_dataclass_payload(item) for item in self._audit_log],
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.path)

    def record_audit(self, *args: Any, **kwargs: Any) -> None:
        super().record_audit(*args, **kwargs)
        self._persist()


def memory_snapshot(memory: Memory) -> dict[str, object]:
    """Return a JSON-like representation for audit events."""

    return {
        "id": memory.id,
        "user_id": memory.user_id,
        "content": memory.content,
        "memory_type": memory.memory_type.value,
        "source_session": memory.source_session,
        "scope": memory.scope.value,
        "source": memory.source.value,
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
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


def memory_stream_snapshot(entry: MemoryStreamEntry) -> dict[str, object]:
    return {
        "id": entry.id,
        "user_id": entry.user_id,
        "content": entry.content,
        "kind": entry.kind.value,
        "importance_score": entry.importance_score,
        "last_accessed_at": entry.last_accessed_at.isoformat(),
        "created_at": entry.created_at.isoformat(),
        "decay_rate": entry.decay_rate,
        "status": entry.status.value,
        "metadata": dict(entry.metadata),
    }


def profile_snapshot(profile: UserProfile) -> dict[str, object]:
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "age": profile.age,
        "occupation": profile.occupation,
        "timezone": profile.timezone,
        "updated_at": profile.updated_at.isoformat(),
    }


def session_state_snapshot(state: SessionState) -> dict[str, object]:
    return {
        "current_topic": state.current_topic,
        "active_entities": list(state.active_entities),
        "open_questions": list(state.open_questions),
        "user_goal": state.user_goal,
        "constraints": list(state.constraints),
        "updated_at": state.updated_at.isoformat(),
    }
