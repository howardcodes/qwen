"""Safety and allowlist checks for agentic tool execution."""
from __future__ import annotations

ALLOWED_TOOLS = {
    "search_memories", "list_open_tasks", "create_or_update_task", "classify_blockers", "generate_next_actions",
    "check_recent_daily_summary_status", "check_celery_job_health", "check_telegram_status", "summarize_recent_activity",
    "write_memory_candidate", "send_telegram_message",
}

def is_tool_allowed(tool_name: str) -> bool:
    return tool_name in ALLOWED_TOOLS

def reject_tool(tool_name: str) -> dict:
    return {"ok": False, "data": {"tool": tool_name}, "error": f"Unknown or disallowed tool: {tool_name}"}
