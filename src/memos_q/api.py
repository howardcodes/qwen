"""FastAPI service for MemOS-Q."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .engine import MemoryOS

memory_os = MemoryOS()


class RememberRequest(BaseModel):
    """Request body for creating a memory."""

    user_id: str
    content: str
    memory_type: str = "episodic"
    source_session: str
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallRequest(BaseModel):
    """Request body for explainable memory recall."""

    user_id: str
    query: str
    query_tags: set[str] = Field(default_factory=set)
    limit: int = 5


app = FastAPI(
    title="MemOS-Q",
    description="Self-evolving memory operating system prototype for AI agents.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""

    return {"status": "ok"}


@app.post("/memories")
def remember(request: RememberRequest) -> dict[str, Any]:
    """Create an auditable memory."""

    memory = memory_os.remember(
        user_id=request.user_id,
        content=request.content,
        memory_type=request.memory_type,
        source_session=request.source_session,
        tags=request.tags,
        metadata=request.metadata,
    )
    return serialize_memory(memory)


@app.post("/recall")
def recall(request: RecallRequest) -> list[dict[str, Any]]:
    """Recall memories with scoring signals and reasoning paths."""

    results = memory_os.recall(
        request.user_id,
        request.query,
        query_tags=request.query_tags,
        limit=request.limit,
    )
    return [
        {
            "memory": serialize_memory(result.memory),
            "score": result.score,
            "explanation": {
                "source_session": result.explanation.source_session,
                "confidence_score": result.explanation.confidence_score,
                "timestamp": result.explanation.timestamp.isoformat(),
                "ranking_signals": result.explanation.ranking_signals,
                "reasoning_path": result.explanation.reasoning_path,
            },
        }
        for result in results
    ]


@app.get("/users/{user_id}/memories")
def inspect(user_id: str, include_inactive: bool = False) -> list[dict[str, Any]]:
    """List memories for user inspection and control."""

    return [serialize_memory(memory) for memory in memory_os.inspect(user_id, include_inactive=include_inactive)]


@app.post("/users/{user_id}/maintenance")
def maintenance(user_id: str) -> dict[str, int]:
    """Run autonomous memory maintenance for a user."""

    return memory_os.maintenance(user_id)


def serialize_memory(memory: Any) -> dict[str, Any]:
    """Serialize a memory for API responses."""

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
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
    }
