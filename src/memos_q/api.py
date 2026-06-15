"""FastAPI backend service for MemOS-Q."""

from __future__ import annotations

import os
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
    return serialize_memory(memory)


@app.post("/recall")
async def recall(request: RecallRequest, user_id: str = Depends(authenticated_user)) -> list[dict[str, Any]]:
    results = await run_in_threadpool(memory_os.recall, user_id, request.query, query_tags=request.query_tags, limit=request.limit)
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
    durable_facts = extract_durable_facts(response)
    saved = []
    for fact in durable_facts:
        saved.append(
            await run_in_threadpool(
                memory_os.remember,
                user_id=user_id,
                content=fact,
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                source_session=request.source_session,
                tags={"agent", "qwen", "extracted"},
                metadata={"source": "agent_extraction"},
                status=MemoryStatus.PENDING_REVIEW,
                actor="qwen-agent",
            )
        )
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
            "rds_postgres_configured": bool(settings.postgres_dsn),
            "redis_url_configured": bool(settings.redis_url),
            "opensearch_vector_engine_configured": bool(settings.opensearch_endpoint),
            "opensearch_index": settings.opensearch_index,
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


@app.delete("/users/me/memories/{memory_id}")
async def delete_memory(memory_id: str, user_id: str = Depends(authenticated_user)) -> dict[str, Any]:
    return serialize_memory(await run_in_threadpool(memory_os.forget, user_id, memory_id, actor="user"))


@app.post("/users/me/maintenance")
async def maintenance(user_id: str = Depends(authenticated_user)) -> dict[str, int]:
    return await run_in_threadpool(memory_os.maintenance, user_id)


def serialize_recall(result: Any) -> dict[str, Any]:
    return {"memory": serialize_memory(result.memory), "score": result.score, "explanation": {"source_session": result.explanation.source_session, "confidence_score": result.explanation.confidence_score, "timestamp": result.explanation.timestamp.isoformat(), "ranking_signals": result.explanation.ranking_signals, "reasoning_path": result.explanation.reasoning_path}}


def serialize_memory(memory: Any) -> dict[str, Any]:
    return {"id": memory.id, "user_id": memory.user_id, "content": memory.content, "memory_type": memory.memory_type.value, "source_session": memory.source_session, "confidence_score": memory.confidence_score, "confidence_reasons": memory.confidence_reasons, "importance_score": memory.importance_score, "novelty_score": memory.novelty_score, "stability_score": memory.stability_score, "status": memory.status.value, "version": memory.version, "tags": sorted(memory.tags), "created_at": memory.created_at.isoformat(), "updated_at": memory.updated_at.isoformat()}


def format_memory_context(recalled: list[Any]) -> str:
    if not recalled:
        return "No prior memories."
    return "\n".join(f"- fact: {item.memory.content}\n  confidence: {item.memory.confidence_score:.2f}\n  source: {item.memory.source_session}\n  updated_at: {item.memory.updated_at.isoformat()}" for item in recalled)


def extract_durable_facts(response: str) -> list[str]:
    """Conservative extraction placeholder that avoids storing raw agent responses."""

    facts = []
    for line in response.splitlines():
        cleaned = line.strip(" -•")
        if cleaned.lower().startswith(("user prefers ", "user uses ", "user is working on ")) and len(cleaned) <= 500:
            facts.append(cleaned)
    return facts[:5]
