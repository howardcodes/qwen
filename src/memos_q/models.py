"""Domain models for MemOS-Q."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class MemoryStatus(StrEnum):
    """Lifecycle state for a memory."""

    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    PENDING_CONFLICT_CONFIRMATION = "pending_conflict_confirmation"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    # Backward-compatible aliases for older rows/tests.
    DEPRECATED = "superseded"
    POSSIBLY_CONFLICTING = "pending_conflict_confirmation"


class MemoryConflictStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"


class MemoryConflictResolution(StrEnum):
    ACCEPTED_CANDIDATE = "accepted_candidate"
    KEPT_EXISTING = "kept_existing"
    MERGED = "merged"
    REJECTED_CANDIDATE = "rejected_candidate"
    MANUAL_RESOLUTION = "manual_resolution"


class MemoryStreamKind(StrEnum):
    """Stanford-style memory stream entry categories."""

    OBSERVATION = "observation"
    REFLECTION = "reflection"
    FACT = "fact"
    PREFERENCE = "preference"
    PROFILE = "profile"


class MemoryType(StrEnum):
    """Supported memory layers and durable production categories."""

    USER_FACT = "user_fact"
    PREFERENCE = "preference"
    TASK = "task"
    PROJECT_CONTEXT = "project_context"
    WORKFLOW = "workflow"
    CONVERSATION_SUMMARY = "conversation_summary"
    SYSTEM_TOOL_EVENT = "system_tool_event"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    OPERATIONAL = "operational"


class RelationType(StrEnum):
    """Relationship types used in the memory graph."""

    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    SUPPORTS = "supports"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


@dataclass(slots=True)
class MemoryStreamEntry:
    """Raw memory stream event retained in Postgres/source-of-truth storage."""

    user_id: str
    content: str
    kind: MemoryStreamKind | str = MemoryStreamKind.OBSERVATION
    importance_score: int = 1
    decay_rate: float = 0.01
    status: MemoryStatus = MemoryStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.kind = MemoryStreamKind(self.kind)
        self.status = MemoryStatus(self.status)
        self.importance_score = max(1, min(10, int(self.importance_score)))
        self.decay_rate = max(0.0, float(self.decay_rate))


@dataclass(slots=True)
class Memory:
    """A single auditable memory item."""

    user_id: str
    content: str
    memory_type: MemoryType | str
    source_session: str
    confidence_score: float = 0.75
    importance_score: float = 0.5
    novelty_score: float = 0.5
    stability_score: float = 0.5
    confidence_reasons: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    sensitivity: str = "low"
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_recalled_at: datetime | None = None
    approved_at: datetime | None = None
    last_confirmed_at: datetime | None = None
    last_seen_at: datetime | None = None
    conflicting_memory_id: str | None = None
    conflict_reason: str | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        self.memory_type = MemoryType(self.memory_type)
        self.status = MemoryStatus(self.status)
        self.confidence_score = clamp_score(self.confidence_score)
        self.importance_score = clamp_score(self.importance_score)
        self.novelty_score = clamp_score(self.novelty_score)
        self.stability_score = clamp_score(self.stability_score)
        self.sensitivity = self.sensitivity.lower()
        self.tags = {tag.lower() for tag in self.tags}


@dataclass(slots=True)
class MemoryConflict:
    user_id: str
    existing_memory_id: str
    candidate_memory_id: str
    conflict_type: str
    existing_content: str
    candidate_content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    status: MemoryConflictStatus = MemoryConflictStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None
    resolution: str | None = None

    def __post_init__(self) -> None:
        self.status = MemoryConflictStatus(self.status)


@dataclass(slots=True)
class MemoryEdge:
    """A typed relationship between two memories."""

    source_memory: str
    target_memory: str
    relation_type: RelationType | str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.relation_type = RelationType(self.relation_type)


@dataclass(slots=True)
class RecallExplanation:
    source_session: str
    confidence_score: float
    timestamp: datetime
    ranking_signals: dict[str, float]
    reasoning_path: list[str]


@dataclass(slots=True)
class RecallResult:
    memory: Memory
    score: float
    explanation: RecallExplanation


@dataclass(slots=True)
class AuditEvent:
    action: str
    actor: str
    memory_id: str
    previous_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    timestamp: datetime = field(default_factory=utc_now)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
