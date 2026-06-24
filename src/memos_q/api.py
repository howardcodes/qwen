"""FastAPI backend service for MemOS-Q."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AnyUrl, BaseModel, Field

from .config import settings
from .engine import MemoryOS
from .integrations.factory import build_memory_store
from .integrations.qwen_cloud import QwenCloudClient, QwenMessage, build_qwen_agent
from .models import MemoryStatus, MemoryType
from .monitoring import memory_metrics as metrics
from .monitoring.observability import add_prometheus_metrics, configure_opentelemetry

qwen_client = QwenCloudClient()
memory_os = MemoryOS(
    build_memory_store(),
    embedding_provider=qwen_client,
    require_live_embeddings=settings.qwen_require_live_embeddings,
    fallback_embedding_dimensions=settings.qwen_embedding_dimensions,
)


class RememberRequest(BaseModel):
    """Request body for creating a memory for the authenticated user."""

    content: str = Field(..., min_length=1, max_length=5000)
    memory_type: MemoryType = MemoryType.USER_FACT
    source_session: str = Field(default="api-session", min_length=1, max_length=128)
    tags: set[str] = Field(default_factory=set, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentChatRequest(BaseModel):
    """Request body for live Qwen-Agent/QwenCloud chat."""

    message: str = Field(..., min_length=1, max_length=8000)
    source_session: str = Field(default="api-session", min_length=1, max_length=128)


class VisionIngestRequest(BaseModel):
    """Request body for Qwen3-VL multimodal ingestion."""

    image_url: AnyUrl
    source_session: str = Field(default="vision-upload", min_length=1, max_length=128)
    prompt: str = Field(default="Extract durable memory facts from this image or document.", min_length=1, max_length=2000)


class MemoryPatchRequest(BaseModel):
    """Request body for editing and approving a memory."""

    content: str = Field(..., min_length=1, max_length=5000)


class ExtractedMemory(BaseModel):
    content: str = Field(..., min_length=1, max_length=500)
    type: MemoryType
    confidence: float = Field(..., ge=0, le=1)
    sensitivity: str = Field(..., pattern="^(low|medium|high)$")
    reason: str = Field(default="", max_length=500)


class RecallRequest(BaseModel):
    """Request body for explainable memory recall."""

    query: str = Field(..., min_length=1, max_length=5000)
    query_tags: set[str] = Field(default_factory=set, max_length=20)
    limit: int = Field(default=5, ge=1, le=20)


def authenticated_user(x_user_id: str = Header(..., min_length=1, max_length=128)) -> str:
    """Resolve the authenticated user from trusted middleware/header context."""

    return x_user_id


app = FastAPI(title="MemOS-Q", description="Self-evolving memory operating system prototype for AI agents.", version="0.1.0")
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


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"resource not found: {exc}"})


@app.exception_handler(requests.RequestException)
async def qwen_error_handler(_: Request, exc: requests.RequestException) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": f"QwenCloud request failed: {exc}"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/memories")
async def remember(request: RememberRequest, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    memory = await run_in_threadpool(
        memory_os.remember,
        user_id=user_id,
        content=request.content,
        memory_type=request.memory_type,
        source_session=request.source_session,
        tags=request.tags,
        metadata={**request.metadata, "source": request.metadata.get("source", "explicit_user")},
        actor="user",
    )
    metrics.memories_created_total.inc()
    refresh_memory_gauges(user_id)
    return serialize_memory(memory)


@app.post("/recall")
async def recall(request: RecallRequest, user_id: str = Depends(authenticated_user)) -> list[dict[str, Any]]:
    start = time.perf_counter()
    results = await run_in_threadpool(memory_os.recall, user_id, request.query, query_tags=request.query_tags, limit=request.limit)
    metrics.memory_search_latency_seconds.observe(time.perf_counter() - start)
    for result in results:
        metrics.memories_recalled_total.inc()
        metrics.memory_recall_score.observe(result.score)
    return [serialize_recall(result) for result in results]


@app.post("/agent/chat")
async def agent_chat(request: AgentChatRequest, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    recalled = await run_in_threadpool(memory_os.recall, user_id, request.message, limit=3)
    memory_context = format_memory_context(recalled)
    response = await run_in_threadpool(
        qwen_client.chat,
        [
            QwenMessage("system", "Use only relevant private memories. Ask before saving sensitive or uncertain facts."),
            QwenMessage("user", f"Recalled memories with provenance:\n{memory_context}\n\nUser: {request.message}"),
        ],
        model=settings.qwen_reasoning_model,
    )
    start = time.perf_counter()
    durable_facts = extract_durable_memories(request.message, response, memory_context)
    metrics.memory_extraction_latency_seconds.observe(time.perf_counter() - start)
    saved = []
    for fact in durable_facts:
        lifecycle_status = status_for_extracted_memory(fact)
        if lifecycle_status is None:
            continue
        saved.append(
            await run_in_threadpool(
                memory_os.remember,
                user_id=user_id,
                content=fact.content,
                memory_type=fact.type,
                source_session=request.source_session,
                tags={"agent", "qwen", "extracted"},
                metadata={"source": "agent_extraction", "reason": fact.reason},
                confidence_score=fact.confidence,
                sensitivity=fact.sensitivity,
                status=lifecycle_status,
                actor="qwen-agent",
            )
        )
        metrics.memories_created_total.inc()
    refresh_memory_gauges(user_id)
    return {"response": response, "recalled_memories": [serialize_memory(item.memory) for item in recalled], "pending_memories": [serialize_memory(item) for item in saved]}


@app.post("/agent/qwen-agent")
async def qwen_agent_chat(request: AgentChatRequest, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    agent = build_qwen_agent()
    chunks = await run_in_threadpool(lambda: list(agent.run(messages=[{"role": "user", "content": request.message}])))
    response = chunks[-1][-1]["content"] if chunks else ""
    memory_os.store.record_audit("tool_call", "qwen-agent", user_id, None, {"tool": "qwen_agent", "session": request.source_session})
    return {"response": response}


@app.post("/ingest/vision")
async def ingest_vision(request: VisionIngestRequest, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    extraction = await run_in_threadpool(qwen_client.vision_extract, image_url=str(request.image_url), prompt=request.prompt)
    memory = await run_in_threadpool(
        memory_os.remember,
        user_id=user_id,
        content=extraction[:5000],
        memory_type=MemoryType.PROJECT_CONTEXT,
        source_session=request.source_session,
        tags={"multimodal", "qwen-vl"},
        metadata={"image_url": str(request.image_url), "source": "agent_extraction"},
        status=MemoryStatus.PENDING_REVIEW,
        actor="qwen-vl-agent",
    )
    return {"memory": serialize_memory(memory), "extraction": extraction}


@app.get("/integrations/status")
async def integrations_status() -> dict[str, Any]:
    return {
        "frontend": {"nextjs_url": settings.frontend_url},
        "backend": {"fastapi": True, "target": "Alibaba Cloud ECS"},
        "storage": {
            "mode": settings.memos_store,
            "postgres_dsn_configured": bool(settings.postgres_dsn),
            "postgres_on_ecs_configured": bool(settings.postgres_dsn),
            "redis_url_configured": bool(settings.redis_url),
            "minio_bucket": settings.s3_bucket,
            "s3_bucket": settings.s3_bucket,
            "pinecone_configured": bool(settings.pinecone_api_key and settings.pinecone_host),
            "pinecone_index": settings.pinecone_index,
            "pinecone_namespace": settings.pinecone_namespace,
        },
        "models": {
            "qwen_api_key_configured": bool(settings.qwen_api_key),
            "base_url": settings.qwen_base_url,
            "embedding_model": settings.qwen_embedding_model,
            "embedding_dimensions": settings.qwen_embedding_dimensions,
            "live_embeddings_required": settings.qwen_require_live_embeddings,
        },
    }


@app.get("/users/me/memories")
async def inspect(include_inactive: bool = False, user_id: str = Depends(authenticated_user)) -> list[dict[str, Any]]:
    memories = await run_in_threadpool(memory_os.inspect, user_id, include_inactive=include_inactive)
    return [serialize_memory(memory) for memory in memories]


@app.post("/users/me/memories/{memory_id}/approve")
async def approve_memory(memory_id: str, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    memory = await run_in_threadpool(memory_os.approve, user_id, memory_id, actor="user")
    metrics.memories_approved_total.inc()
    refresh_memory_gauges(user_id)
    return serialize_memory(memory)


@app.post("/users/me/memories/{memory_id}/reject")
async def reject_memory(memory_id: str, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    memory = await run_in_threadpool(memory_os.reject, user_id, memory_id, actor="user")
    metrics.memories_rejected_total.inc()
    refresh_memory_gauges(user_id)
    return serialize_memory(memory)


@app.patch("/users/me/memories/{memory_id}")
async def edit_memory(memory_id: str, request: MemoryPatchRequest, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    memory = await run_in_threadpool(memory_os.edit_and_approve, user_id, memory_id, content=request.content, actor="user")
    metrics.memories_approved_total.inc()
    refresh_memory_gauges(user_id)
    return serialize_memory(memory)


@app.delete("/users/me/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    memory = await run_in_threadpool(memory_os.forget, user_id, memory_id, actor="user")
    metrics.memories_deleted_total.inc()
    refresh_memory_gauges(user_id)
    return serialize_memory(memory)


@app.post("/users/me/maintenance")
async def maintenance(user_id: str = Depends(authenticated_user)) -> dict[str, int]:
    return await run_in_threadpool(memory_os.maintenance, user_id)


def serialize_recall(result: Any) -> dict[str, Any]:
    return {"memory": serialize_memory(result.memory), "score": result.score, "explanation": {"source_session": result.explanation.source_session, "confidence_score": result.explanation.confidence_score, "timestamp": result.explanation.timestamp.isoformat(), "ranking_signals": result.explanation.ranking_signals, "reasoning_path": result.explanation.reasoning_path}}


def serialize_memory(memory: Any) -> dict[str, Any]:
    return {"id": memory.id, "user_id": memory.user_id, "content": memory.content, "memory_type": memory.memory_type.value, "source_session": memory.source_session, "confidence_score": memory.confidence_score, "confidence_reasons": memory.confidence_reasons, "importance_score": memory.importance_score, "novelty_score": memory.novelty_score, "stability_score": memory.stability_score, "status": memory.status.value, "sensitivity": memory.sensitivity, "approved_at": memory.approved_at.isoformat() if memory.approved_at else None, "last_seen_at": memory.last_seen_at.isoformat() if memory.last_seen_at else None, "conflicting_memory_id": memory.conflicting_memory_id, "conflict_reason": memory.conflict_reason, "version": memory.version, "tags": sorted(memory.tags), "created_at": memory.created_at.isoformat(), "updated_at": memory.updated_at.isoformat()}


def format_memory_context(recalled: list[Any]) -> str:
    if not recalled:
        return "No prior memories."
    return "\n".join(f"- fact: {item.memory.content}\n  confidence: {item.memory.confidence_score:.2f}\n  source: {item.memory.source_session}\n  updated_at: {item.memory.updated_at.isoformat()}" for item in recalled)


def status_for_extracted_memory(memory: ExtractedMemory) -> MemoryStatus | None:
    if memory.confidence < 0.60:
        return None
    if memory.confidence >= 0.90 and memory.sensitivity == "low":
        return MemoryStatus.ACTIVE
    return MemoryStatus.PENDING_REVIEW


def extract_durable_memories(user_message: str, assistant_response: str, conversation_context: str) -> list[ExtractedMemory]:
    """Ask Qwen for strict JSON memories using message, response, and context."""

    prompt = (
        "Extract durable memories from the user message, assistant response, and context. "
        "Return only a JSON array of objects with content, type, confidence, sensitivity, reason. "
        "Allowed types: preference, user_fact, project_context, task, workflow. "
        "Allowed sensitivity: low, medium, high.\n"
        f"Context:\n{conversation_context}\nUser Message:\n{user_message}\nAssistant Response:\n{assistant_response}"
    )
    try:
        raw = qwen_client.chat([QwenMessage("system", "Return strict JSON only."), QwenMessage("user", prompt)], temperature=0)
        data = json.loads(raw)
        return [ExtractedMemory.model_validate(item) for item in data][:5]
    except Exception:
        metrics.qwen_errors_total.inc()
        return []


def refresh_memory_gauges(user_id: str) -> None:
    memories = memory_os.inspect(user_id, include_inactive=True)
    metrics.active_memory_count.set(sum(1 for item in memories if item.status == MemoryStatus.ACTIVE))
    metrics.pending_review_count.set(sum(1 for item in memories if item.status == MemoryStatus.PENDING_REVIEW))
    metrics.conflicting_memory_count.set(sum(1 for item in memories if item.status == MemoryStatus.POSSIBLY_CONFLICTING))
