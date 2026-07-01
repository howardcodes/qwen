"""Routing predicates for the agentic LangGraph workflow."""
from __future__ import annotations
from typing import Any

def route_after_planner(state: dict[str, Any]) -> str:
    if not state.get("plan") or state.get("metadata", {}).get("tool_calls", 0) >= 8:
        return "finalize"
    return "tool_executor"

def route_after_reflection(state: dict[str, Any]) -> str:
    meta = state.get("metadata", {})
    if meta.get("needs_replan") and meta.get("planning_rounds", 0) < 2:
        return "planner"
    return "update_memory_and_tasks"
