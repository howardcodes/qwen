"""Daily proactive summary generation for MemOS-Q."""

from __future__ import annotations

import logging
from datetime import timedelta

from .config import settings
from .integrations.qwen_cloud import QwenCloudClient, QwenMessage
from .models import DailySummary, DailySummarySettings, MemoryStatus, utc_now
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


def generate_daily_summary_text(user_id: str, store, qwen: QwenCloudClient) -> str:
    since = utc_now() - timedelta(hours=24)
    conversations = store.recent_conversation_activity(user_id, since=since, limit=80) if hasattr(store, "recent_conversation_activity") else []
    memories = [m for m in store.list_memories(user_id) if m.status == MemoryStatus.ACTIVE]
    important = sorted(memories, key=lambda m: (m.updated_at, m.importance_score), reverse=True)[:20]
    if not conversations and not important:
        return ""
    prompt = (
        "Create a concise Telegram-readable daily summary for the user. Use headings and short bullets. "
        "Include: topics discussed, key decisions/preferences learned, important memories updated, suggested related follow-up topics or next steps. "
        "Deduplicate overlapping topics. Do not invent facts. If a section has nothing useful, say 'None'. Keep under 1200 characters.\n\n"
        "Recent conversations from the last 24 hours:\n"
        + ("\n".join(f"- {item['role']}: {item['content']}" for item in conversations) or "- None")
        + "\n\nRelevant long-term memories:\n"
        + ("\n".join(f"- {m.content}" for m in important) or "- None")
    )
    raw = qwen.chat(
        [QwenMessage("system", "You generate concise proactive memory summaries."), QwenMessage("user", prompt[:6000])],
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
