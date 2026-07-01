"""Qwen-backed task planning and decomposition."""
from __future__ import annotations
import json
from typing import Any
from .prompts import PLANNER_SYSTEM_PROMPT
from .safety import ALLOWED_TOOLS
from .llm import QwenChatHelper

def default_plan(state: dict[str, Any]) -> dict[str, Any]:
    trigger = state.get("trigger")
    plan = [{"step_id": "step_1", "tool": "list_open_tasks", "reason": "Load durable open tasks", "args": {}}]
    if trigger in {"telegram_daily_briefing", "system_health_check"}:
        plan += [{"step_id": "step_2", "tool": "check_recent_daily_summary_status", "reason": "Check briefing history", "args": {}}, {"step_id": "step_3", "tool": "check_telegram_status", "reason": "Check Telegram readiness", "args": {}}, {"step_id": "step_4", "tool": "summarize_recent_activity", "reason": "Summarize recent activity", "args": {}}]
    elif state.get("input_text"):
        plan.insert(0, {"step_id": "step_0", "tool": "search_memories", "reason": "Find relevant memories", "args": {"query": state.get("input_text", "")}})
    return {"goal": "Produce an evidence-based agent briefing", "plan": plan, "success_criteria": ["Evidence gathered", "Briefing is actionable"], "risk_level": "low"}

def plan_agent_run(state: dict[str, Any], qwen_client: Any) -> dict[str, Any]:
    meta = state.setdefault("metadata", {})
    meta["planning_rounds"] = meta.get("planning_rounds", 0) + 1
    fallback = default_plan(state)
    prompt = json.dumps({"state": {k: state.get(k) for k in ["trigger", "input_text", "recent_conversations", "open_tasks", "system_health"]}, "allowed_tools": sorted(ALLOWED_TOOLS), "schema": fallback}, default=str)[:7000]
    data = QwenChatHelper(qwen_client).json(PLANNER_SYSTEM_PROMPT, prompt, fallback)
    for err in data.get("errors", []):
        state.setdefault("errors", []).append(err)
    seen=set(); clean=[]
    for i, step in enumerate(data.get("plan", [])):
        tool = str(step.get("tool", "")); args = step.get("args") if isinstance(step.get("args"), dict) else {}
        key=(tool, json.dumps(args, sort_keys=True))
        if tool in ALLOWED_TOOLS and key not in seen:
            seen.add(key); clean.append({"step_id": str(step.get("step_id") or f"step_{i+1}"), "tool": tool, "reason": str(step.get("reason", "")), "args": args})
    state["plan"] = clean[: max(0, 8 - meta.get("tool_calls", 0))]
    return state
