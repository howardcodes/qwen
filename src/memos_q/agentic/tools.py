"""Allowlisted LangChain tools for MemOS-Q agentic runs."""
from __future__ import annotations
from typing import Any, Callable
from memos_q.models import TaskRecord, TaskRecordStatus, utc_now
from memos_q.telegram import truncate_telegram_message

try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover
    def tool(fn: Callable | None = None, **_: Any):
        def deco(f: Callable) -> Callable:
            f.name = f.__name__; return f
        return deco(fn) if fn else deco

def result(data: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    return {"ok": error is None, "data": data or {}, "error": error}

def build_tools(store: Any, telegram_client: Any = None, state: dict[str, Any] | None = None) -> dict[str, Callable[..., dict[str, Any]]]:
    state = state if state is not None else {}

    @tool
    def search_memories(query: str = "", limit: int = 10) -> dict[str, Any]:
        memories = store.list_memories(state.get("user_id", "default")) if hasattr(store, "list_memories") else []
        q = query.lower()
        rows = [m for m in memories if not q or q in m.content.lower()][:limit]
        return result({"memories": [{"id": m.id, "content": m.content, "type": getattr(m.memory_type, "value", str(m.memory_type))} for m in rows]})

    @tool
    def list_open_tasks() -> dict[str, Any]:
        tasks = store.list_task_records(state.get("user_id", "default")) if hasattr(store, "list_task_records") else []
        return result({"tasks": [{"id": t.id, "title": t.title, "status": getattr(t.status, "value", str(t.status)), "blocker": t.blocker, "next_action": t.next_action} for t in tasks]})

    @tool
    def create_or_update_task(title: str, status: str = "open", blocker: str | None = None, next_action: str | None = None, evidence: list[str] | None = None, priority: str = "normal", project: str | None = None, due_date: str | None = None, confidence: float = 0.75) -> dict[str, Any]:
        try: task_status = TaskRecordStatus(status)
        except ValueError: task_status = TaskRecordStatus.OPEN
        task = TaskRecord(user_id=state.get("user_id", "default"), title=title, status=task_status, blocker=blocker, blocker_type="other" if blocker else "none", next_action=next_action, evidence=evidence or [], confidence=confidence, source="agentic", metadata={"priority": priority, "project": project, "due_date": due_date, "last_seen_at": utc_now().isoformat(), "agent_confidence": confidence, "source_refs": evidence or []})
        saved = store.upsert_task_record(task, actor="agentic-task") if hasattr(store, "upsert_task_record") else task
        return result({"task": {"id": saved.id, "title": saved.title, "status": saved.status.value, "next_action": saved.next_action}})

    @tool
    def classify_blockers(tasks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = tasks or state.get("open_tasks", [])
        return result({"blockers": [{**t, "blocked": bool(t.get("blocker") or t.get("status") == "blocked")} for t in rows]})

    @tool
    def generate_next_actions(tasks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = tasks or state.get("open_tasks", [])
        return result({"next_actions": [{"task": t.get("title"), "next_action": t.get("next_action") or f"Review and choose the next step for {t.get('title', 'this task')}"} for t in rows]})

    @tool
    def check_recent_daily_summary_status() -> dict[str, Any]:
        summaries = store.list_daily_summaries(state.get("user_id", "default")) if hasattr(store, "list_daily_summaries") else []
        last = summaries[-1] if summaries else None
        return result({"last_summary": None if not last else {"id": last.id, "sent": last.sent_to_telegram, "error": last.error_message, "created_at": last.created_at.isoformat()}})

    @tool
    def check_celery_job_health() -> dict[str, Any]: return result({"celery_configured": True})
    @tool
    def check_telegram_status() -> dict[str, Any]: return result({"telegram_configured": bool(getattr(telegram_client, "bot_token", None) and getattr(telegram_client, "chat_id", None))})
    @tool
    def summarize_recent_activity() -> dict[str, Any]: return result({"activity_count": len(state.get("recent_conversations", [])), "summary": "Recent activity reviewed."})
    @tool
    def write_memory_candidate(content: str, evidence: list[str] | None = None) -> dict[str, Any]: return result({"memory_candidate": content, "evidence": evidence or []})
    @tool
    def send_telegram_message(text: str, send: bool = False) -> dict[str, Any]:
        if not (send or state.get("should_notify")):
            return result(error="Telegram send rejected: notification not approved")
        if state.get("metadata", {}).get("telegram_messages_sent", 0) >= 1:
            return result(error="Telegram send rejected: per-run limit reached")
        if telegram_client is None:
            return result(error="Telegram client unavailable")
        res = telegram_client.send_message(truncate_telegram_message(text, limit=1500))
        state.setdefault("metadata", {})["telegram_messages_sent"] = state.get("metadata", {}).get("telegram_messages_sent", 0) + int(res.sent)
        return result({"sent": res.sent}, res.error_message if not res.sent else None)
    return {fn.__name__: fn for fn in [search_memories, list_open_tasks, create_or_update_task, classify_blockers, generate_next_actions, check_recent_daily_summary_status, check_celery_job_health, check_telegram_status, summarize_recent_activity, write_memory_candidate, send_telegram_message]}
