"""Conditional Telegram notification policy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(slots=True)
class NotificationDecision:
    should_notify: bool
    reason: str

def decide_notification(state: dict[str, Any]) -> NotificationDecision:
    meta = state.get("metadata", {})
    if meta.get("force_send") or meta.get("send"):
        return NotificationDecision(True, "force_send requested")
    text = (state.get("input_text") or "").lower()
    if "summary" in text or "briefing" in text:
        return NotificationDecision(True, "user explicitly requested summary")
    if any(t.get("status") == "blocked" or t.get("blocker") for t in state.get("open_tasks", []) + state.get("task_updates", [])):
        return NotificationDecision(True, "blocked task detected")
    if any(u.get("change") in {"created", "status_changed"} for u in state.get("task_updates", [])):
        return NotificationDecision(True, "important task update detected")
    health = state.get("system_health", {})
    if any(v is False for v in health.values() if isinstance(v, bool)) or health.get("errors"):
        return NotificationDecision(True, "system failure detected")
    if state.get("open_tasks") and any(t.get("next_action") for t in state.get("open_tasks", [])):
        return NotificationDecision(True, "concrete next action available")
    if not state.get("recent_conversations") and not state.get("open_tasks"):
        return NotificationDecision(False, "Skipped: no actionable updates")
    if state.get("metadata", {}).get("confidence", 1.0) < 0.45:
        return NotificationDecision(False, "Skipped: confidence too low")
    return NotificationDecision(False, "Skipped: no meaningful changes")
