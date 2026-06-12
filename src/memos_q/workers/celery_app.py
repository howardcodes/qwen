"""Celery worker entrypoint for autonomous memory maintenance."""

from __future__ import annotations

from celery import Celery

from memos_q.config import settings
from memos_q.engine import MemoryOS
from memos_q.integrations.qwen_cloud import QwenCloudClient, QwenMessage

celery_app = Celery(
    "memos_q",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    beat_schedule={
        "nightly-memory-compaction": {
            "task": "memos_q.workers.celery_app.compact_user_memories",
            "schedule": 60 * 60 * 24,
            "args": ("default-user",),
        }
    },
)

_memory_os = MemoryOS()
_qwen = QwenCloudClient()


@celery_app.task(name="memos_q.workers.celery_app.compact_user_memories")
def compact_user_memories(user_id: str) -> dict[str, int]:
    """Run duplicate merging, promotion, decay, and archival asynchronously."""

    return _memory_os.maintenance(user_id)


@celery_app.task(name="memos_q.workers.celery_app.summarize_session")
def summarize_session(user_id: str, source_session: str, transcript: str) -> str:
    """Summarize an old session with Qwen3.5-Flash and store it as memory."""

    summary = _qwen.chat(
        [
            QwenMessage("system", "Summarize the session into durable, auditable memory facts."),
            QwenMessage("user", transcript),
        ],
        model=settings.qwen_flash_model,
        temperature=0,
    )
    _memory_os.remember(
        user_id=user_id,
        content=summary,
        memory_type="episodic",
        source_session=source_session,
        tags={"summary", "celery"},
        actor="celery-compaction-agent",
    )
    return summary
