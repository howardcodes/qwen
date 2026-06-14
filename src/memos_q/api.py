"""FastAPI service for MemOS-Q."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .engine import MemoryOS
from .integrations.factory import build_memory_store
from .integrations.qwen_cloud import QwenCloudClient, QwenMessage, build_qwen_agent
from .monitoring.observability import add_prometheus_metrics, configure_opentelemetry

memory_os = MemoryOS(build_memory_store())
qwen_client = QwenCloudClient()


class RememberRequest(BaseModel):
    """Request body for creating a memory."""

    user_id: str
    content: str
    memory_type: str = "episodic"
    source_session: str
    tags: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentChatRequest(BaseModel):
    """Request body for live Qwen-Agent/QwenCloud chat."""

    user_id: str
    message: str
    source_session: str = "api-session"


class VisionIngestRequest(BaseModel):
    """Request body for Qwen3-VL multimodal ingestion."""

    user_id: str
    image_url: str
    source_session: str = "vision-upload"
    prompt: str = "Extract durable memory facts from this image or document."


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
add_prometheus_metrics(app)
if os.getenv("MEMOS_ENABLE_OTEL", "false").lower() == "true":
    configure_opentelemetry(app)


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


@app.post("/agent/chat")
def agent_chat(request: AgentChatRequest) -> dict[str, Any]:
    """Run a live QwenCloud-backed agent turn and write the result to memory."""

    recalled = memory_os.recall(request.user_id, request.message, limit=3)
    memory_context = "\n".join(f"- {item.memory.content}" for item in recalled) or "No prior memories."
    response = qwen_client.chat(
        [
            QwenMessage("system", "You are MemOS-Q. Use recalled memories transparently."),
            QwenMessage("user", f"Recalled memories:\n{memory_context}\n\nUser: {request.message}"),
        ],
        model=settings.qwen_reasoning_model,
    )
    memory_os.remember(
        user_id=request.user_id,
        content=f"Agent response: {response}",
        memory_type="operational",
        source_session=request.source_session,
        tags={"agent", "qwen"},
        actor="qwen-agent",
    )
    return {"response": response, "recalled_memories": [serialize_memory(item.memory) for item in recalled]}


@app.post("/agent/qwen-agent")
def qwen_agent_chat(request: AgentChatRequest) -> dict[str, Any]:
    """Run a turn through the Qwen-Agent Assistant integration."""

    agent = build_qwen_agent()
    messages = [{"role": "user", "content": request.message}]
    chunks = list(agent.run(messages=messages))
    response = chunks[-1][-1]["content"] if chunks else ""
    memory_os.remember(
        user_id=request.user_id,
        content=f"Qwen-Agent response: {response}",
        memory_type="operational",
        source_session=request.source_session,
        tags={"qwen-agent"},
        actor="qwen-agent",
    )
    return {"response": response}


@app.post("/ingest/vision")
def ingest_vision(request: VisionIngestRequest) -> dict[str, Any]:
    """Use Qwen3-VL to extract memories from an image/PDF URL."""

    extraction = qwen_client.vision_extract(image_url=request.image_url, prompt=request.prompt)
    memory = memory_os.remember(
        user_id=request.user_id,
        content=extraction,
        memory_type="episodic",
        source_session=request.source_session,
        tags={"multimodal", "qwen-vl"},
        metadata={"image_url": request.image_url},
        actor="qwen-vl-agent",
    )
    return {"memory": serialize_memory(memory), "extraction": extraction}


@app.get("/integrations/status")
def integrations_status() -> dict[str, Any]:
    """Show which live integrations have credentials or endpoints configured."""

    return {
        "frontend": {"nextjs_url": settings.frontend_url},
        "backend": {"fastapi": True, "qwen_agent_available": bool(settings.qwen_api_key)},
        "models": {
            "qwen_api_key_configured": bool(settings.qwen_api_key),
            "reasoning_model": settings.qwen_reasoning_model,
            "flash_model": settings.qwen_flash_model,
            "vision_model": settings.qwen_vl_model,
        },
        "storage": {
            "postgres_dsn_configured": bool(settings.postgres_dsn),
            "redis_url_configured": bool(settings.redis_url),
            "s3_bucket": settings.s3_bucket,
        },
        "jobs": {"celery_broker_url_configured": bool(settings.celery_broker_url)},
        "monitoring": {
            "langfuse_configured": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
            "otel_endpoint": settings.otel_exporter_otlp_endpoint,
            "prometheus_metrics_path": "/metrics",
        },
    }


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
