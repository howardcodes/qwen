"""Celery worker entrypoint for autonomous memory maintenance."""

from __future__ import annotations

from celery import Celery

from memos_q.config import settings
from memos_q.engine import MemoryOS
from memos_q.integrations.factory import build_memory_store
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
    task_acks_late=True,
    task_default_retry_delay=60,
    task_routes={"memos_q.workers.celery_app.*": {"queue": "memory-maintenance"}},
    beat_schedule={
        "nightly-memory-compaction": {
            "task": "memos_q.workers.celery_app.compact_all_active_users",
            "schedule": 60 * 60 * 24,
        }
    },
)

_memory_os = MemoryOS(build_memory_store())
_qwen = QwenCloudClient()


@celery_app.task(name="memos_q.workers.celery_app.compact_all_active_users")
def compact_all_active_users() -> dict[str, str]:
    """Run scheduled memory maintenance for every active user in batches."""

    user_ids = _memory_os.store.list_user_ids()
    return {user_id: compact_user_memories.delay(user_id).id for user_id in user_ids}


@celery_app.task(
    bind=True,
    name="memos_q.workers.celery_app.compact_user_memories",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def compact_user_memories(self, user_id: str) -> dict[str, int]:
    """Run duplicate merging, promotion, decay, and archival asynchronously."""

    _memory_os.store.record_audit("job_start", "celery-maintenance", user_id, None, {"task_id": self.request.id})
    report = _memory_os.maintenance(user_id)
    _memory_os.store.record_audit("job_success", "celery-maintenance", user_id, None, {"task_id": self.request.id, "report": report})
    return report


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
        memory_type="conversation_summary",
        source_session=source_session,
        tags={"summary", "celery"},
        metadata={"source": "agent_extraction"},
        actor="celery-compaction-agent",
    )
    return summary
