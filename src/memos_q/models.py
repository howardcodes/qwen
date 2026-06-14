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
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class MemoryType(StrEnum):
    """Supported memory layers."""

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
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_recalled_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        self.memory_type = MemoryType(self.memory_type)
        self.confidence_score = clamp_score(self.confidence_score)
        self.importance_score = clamp_score(self.importance_score)
        self.novelty_score = clamp_score(self.novelty_score)
        self.stability_score = clamp_score(self.stability_score)
        self.tags = {tag.lower() for tag in self.tags}


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
    """Human-readable provenance and ranking details for a recalled memory."""

    source_session: str
    confidence_score: float
    timestamp: datetime
    ranking_signals: dict[str, float]
    reasoning_path: list[str]


@dataclass(slots=True)
class RecallResult:
    """A recalled memory plus its final score and explanation."""

    memory: Memory
    score: float
    explanation: RecallExplanation


@dataclass(slots=True)
class AuditEvent:
    """Immutable audit event for memory lifecycle changes."""

    action: str
    actor: str
    memory_id: str
    previous_value: dict[str, Any] | None
    new_value: dict[str, Any] | None
    timestamp: datetime = field(default_factory=utc_now)


def clamp_score(value: float) -> float:
    """Clamp a score to the inclusive [0, 1] range."""

    return max(0.0, min(1.0, float(value)))
