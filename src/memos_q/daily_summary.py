"""Daily proactive summary and agentic briefing generation for MemOS-Q."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from .config import settings
from .integrations.qwen_cloud import QwenCloudClient, QwenMessage
from .models import DailySummary, DailySummarySettings, MemoryStatus, TaskRecord, TaskRecordStatus, utc_now
from .telegram import TelegramClient, truncate_telegram_message

logger = logging.getLogger(__name__)


def deduplicate_lines(text: str) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for line in text.splitlines():
        key = line.strip().lower().lstrip("-*•0123456789. )")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _active_memories(store: Any, user_id: str):
    return [m for m in store.list_memories(user_id) if m.status == MemoryStatus.ACTIVE]


def _recent_inputs(user_id: str, store: Any) -> tuple[list[dict[str, str]], list[Any], list[TaskRecord]]:
    since = utc_now() - timedelta(hours=24)
    conversations = store.recent_conversation_activity(user_id, since=since, limit=80) if hasattr(store, "recent_conversation_activity") else []
    memories = _active_memories(store, user_id)
    important = sorted(memories, key=lambda m: (m.updated_at, m.importance_score), reverse=True)[:20]
    tasks = store.list_task_records(user_id)[:20] if hasattr(store, "list_task_records") else []
    return conversations, important, tasks


def _json_array(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.lower().startswith("json") else raw
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end >= start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def extract_and_persist_task_records(user_id: str, store: Any, qwen: QwenCloudClient) -> list[TaskRecord]:
    """Detect open goals/tasks from recent conversation and memory, then persist durable task state."""

    conversations, important, existing = _recent_inputs(user_id, store)
    if not conversations and not important and not existing:
        return []
    prompt = (
        "Extract durable open goals/tasks for the user from the supplied recent conversations, memories, and existing task records. "
        "Use generic reasoning only; do not depend on any specific example. Return strict JSON array only. "
        "Each object must include: title, status, blocker_type, blocker, next_action, evidence, confidence. "
        "Allowed status values: open, blocked, in_progress, done, dropped. "
        "Allowed blocker_type values: none, waiting_on_user, waiting_on_external, missing_info, technical, scheduling, decision_needed, resource, other. "
        "Only include tasks/goals supported by evidence. Merge duplicates by using a stable concise title. "
        "Mark done/dropped only when explicitly supported. Suggest exactly one concrete next_action when possible.\n\n"
        "Recent conversations from the last 24 hours:\n"
        + ("\n".join(f"- {item['role']}: {item['content']}" for item in conversations) or "- None")
        + "\n\nRelevant long-term memories:\n"
        + ("\n".join(f"- {m.content}" for m in important) or "- None")
        + "\n\nExisting durable task records:\n"
        + ("\n".join(f"- {t.title} [{t.status.value}]; blocker={t.blocker or 'none'}; next={t.next_action or 'none'}" for t in existing) or "- None")
    )
    raw = qwen.chat(
        [QwenMessage("system", "You are a task-state extraction agent. Return strict JSON only."), QwenMessage("user", prompt[:7000])],
        model=settings.qwen_flash_model,
        temperature=0,
        max_tokens=settings.qwen_memory_extraction_max_tokens,
    )
    saved: list[TaskRecord] = []
    for item in _json_array(raw)[:10]:
        title = str(item.get("title", "")).strip()
        confidence = float(item.get("confidence", 0))
        if not title or confidence < 0.55:
            continue
        try:
            status = TaskRecordStatus(str(item.get("status", "open")))
        except ValueError:
            status = TaskRecordStatus.OPEN
        task = TaskRecord(
            user_id=user_id,
            title=title,
            status=status,
            blocker_type=str(item.get("blocker_type") or "none"),
            blocker=str(item["blocker"]).strip() if item.get("blocker") else None,
            next_action=str(item["next_action"]).strip() if item.get("next_action") else None,
            evidence=[str(e) for e in item.get("evidence", [])][:5] if isinstance(item.get("evidence", []), list) else [],
            confidence=confidence,
            source="daily_briefing",
            metadata={"extracted_at": utc_now().isoformat()},
        )
        saved.append(store.upsert_task_record(task) if hasattr(store, "upsert_task_record") else task)
    return saved


def generate_daily_summary_text(user_id: str, store, qwen: QwenCloudClient) -> str:
    conversations, important, tasks = _recent_inputs(user_id, store)
    extracted = extract_and_persist_task_records(user_id, store, qwen) if hasattr(store, "upsert_task_record") else []
    tasks = store.list_task_records(user_id)[:20] if hasattr(store, "list_task_records") else extracted
    if not conversations and not important and not tasks:
        return ""
    prompt = (
        "Create a concise Telegram-readable agentic daily briefing for the user. Use headings and short bullets. "
        "Include: key updates from recent conversations, open goals/tasks, blocker classification, and suggested next actions. "
        "Use the durable task records as source of truth for task state when available. Deduplicate overlapping topics. "
        "Do not invent facts. If a section has nothing useful, say 'None'. Keep under 1200 characters.\n\n"
        "Recent conversations from the last 24 hours:\n"
        + ("\n".join(f"- {item['role']}: {item['content']}" for item in conversations) or "- None")
        + "\n\nRelevant long-term memories:\n"
        + ("\n".join(f"- {m.content}" for m in important) or "- None")
        + "\n\nDurable task records:\n"
        + ("\n".join(f"- {t.title} [{t.status.value}] blocker={t.blocker_type}: {t.blocker or 'none'} next={t.next_action or 'none'}" for t in tasks) or "- None")
    )
    raw = qwen.chat(
        [QwenMessage("system", "You generate concise proactive agentic daily briefings."), QwenMessage("user", prompt[:7000])],
        model=settings.qwen_flash_model,
        temperature=0,
        max_tokens=settings.qwen_summary_max_tokens,
    )
    return truncate_telegram_message(deduplicate_lines(raw), limit=3900)


def run_daily_summary(user_id: str, store, qwen: QwenCloudClient, telegram: TelegramClient | None = None, *, force_send: bool = True) -> DailySummary:
    user_settings = store.get_daily_summary_settings(user_id) if hasattr(store, "get_daily_summary_settings") else DailySummarySettings(user_id=user_id)
    summary = DailySummary(user_id=user_id)
    telegram = telegram or TelegramClient(chat_id=user_settings.telegram_chat_id or None)
    try:
        if not force_send and not user_settings.enabled:
            summary.error_message = "Daily Telegram summary is disabled for this user"
            return store.add_daily_summary(summary) if hasattr(store, "add_daily_summary") else summary
        summary_text = generate_daily_summary_text(user_id, store, qwen)
        if not summary_text.strip():
            summary.error_message = "Summary was empty; nothing sent"
            return store.add_daily_summary(summary) if hasattr(store, "add_daily_summary") else summary
        summary.summary_text = summary_text
        result = telegram.send_message(summary_text, chat_id=user_settings.telegram_chat_id or None)
        summary.sent_to_telegram = result.sent
        summary.error_message = result.error_message
        if result.sent:
            summary.sent_at = utc_now()
    except Exception as exc:  # worker robustness: persist error and do not crash callers
        logger.exception("Daily summary generation failed for user %s", user_id)
        summary.error_message = str(exc)
    return store.add_daily_summary(summary) if hasattr(store, "add_daily_summary") else summary
