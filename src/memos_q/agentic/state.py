"""LangGraph state schema for MemOS-Q agentic runs."""
from __future__ import annotations
from typing import Any, Literal, TypedDict

AgentTrigger = Literal["telegram_daily_briefing", "manual_api_run", "incoming_user_message", "memory_reflection", "system_health_check"]

class AgentState(TypedDict, total=False):
    user_id: str
    trigger: AgentTrigger | str
    input_text: str | None
    recent_conversations: list[dict[str, Any]]
    relevant_memories: list[dict[str, Any]]
    open_tasks: list[dict[str, Any]]
    system_health: dict[str, Any]
    plan: list[dict[str, Any]]
    current_step_index: int
    observations: list[dict[str, Any]]
    task_updates: list[dict[str, Any]]
    final_briefing: str | None
    should_notify: bool
    notification_reason: str | None
    errors: list[str]
    metadata: dict[str, Any]
