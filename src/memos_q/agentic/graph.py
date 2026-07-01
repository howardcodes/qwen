"""LangGraph workflow for MemOS-Q agentic runs."""
from __future__ import annotations
import logging, time
from typing import Any
from uuid import uuid4
from .memory import load_agent_context, retrieve_relevant_memories, persist_memory_updates
from .notification import decide_notification
from .planner import plan_agent_run
from .reflection import reflect_on_run
from .router import route_after_planner, route_after_reflection
from .safety import is_tool_allowed, reject_tool
from .tools import build_tools
from memos_q.models import AgentRun, DailySummary, utc_now
from memos_q.telegram import truncate_telegram_message
from memos_q.monitoring import agent_metrics
logger = logging.getLogger(__name__)

class SimpleCompiledGraph:
    def __init__(self, nodes): self.nodes = nodes
    def invoke(self, state):
        for n in self.nodes: state = n(state)
        return state

def _log(event: str, **kw): logger.info(event, extra={"agent": kw})

def build_agent_graph(store: Any, qwen_client: Any, telegram_client: Any = None):
    def node(name):
        def deco(fn):
            def wrapped(state):
                _log("agent.node.started", node=name, run_id=state.get("metadata",{}).get("run_id"));
                try: out=fn(state); _log("agent.node.completed", node=name); return out
                except Exception as exc:
                    logger.exception("agent.node.failed"); state.setdefault("errors", []).append(f"{name}: {exc}"); return state
            return wrapped
        return deco
    @node("load_context")
    def load_context(state): state.update(load_agent_context(state["user_id"], store)); return state
    @node("retrieve_memory")
    def retrieve_memory(state): state["relevant_memories"] = retrieve_relevant_memories(state["user_id"], state.get("input_text"), store); return state
    @node("load_tasks")
    def load_tasks(state): state["open_tasks"] = load_agent_context(state["user_id"], store).get("open_tasks", []); return state
    @node("check_system_health")
    def health(state): state["system_health"] = {"store": True, "qwen_client": qwen_client is not None, "telegram_client": telegram_client is not None}; return state
    @node("planner")
    def planner(state): return plan_agent_run(state, qwen_client)
    @node("tool_executor")
    def tools_node(state):
        tools = build_tools(store, telegram_client, state)
        for step in state.get("plan", []):
            if state.setdefault("metadata", {}).get("tool_calls", 0) >= 8: break
            name, args = step.get("tool"), step.get("args", {}) or {}
            _log("agent.tool.called", tool=name); agent_metrics.agent_tool_calls_total.inc()
            if not is_tool_allowed(str(name)) or name not in tools:
                obs = {"step_id": step.get("step_id"), **reject_tool(str(name))}; agent_metrics.agent_tool_failures_total.inc(); _log("agent.tool.failed", tool=name)
            else:
                try: obs = {"step_id": step.get("step_id"), "tool": name, **tools[name](**args)}
                except Exception as exc: obs = {"step_id": step.get("step_id"), "tool": name, "ok": False, "data": {}, "error": str(exc)}; agent_metrics.agent_tool_failures_total.inc(); _log("agent.tool.failed", tool=name)
            state.setdefault("observations", []).append(obs); state["metadata"]["tool_calls"] = state["metadata"].get("tool_calls", 0) + 1
        return state
    @node("reflect")
    def reflect(state):
        before=state.get("metadata",{}).get("replans",0); state=reflect_on_run(state, qwen_client)
        if state.get("metadata",{}).get("replans",0)>before: agent_metrics.agent_replans_total.inc(); _log("agent.replan.triggered")
        return state
    @node("update_memory_and_tasks")
    def update(state): state.setdefault("metadata", {})["memory_updates"] = persist_memory_updates(state, store); return state
    @node("notification_decision")
    def notify(state):
        d=decide_notification(state); state["should_notify"]=d.should_notify; state["notification_reason"]=d.reason
        (agent_metrics.agent_notifications_sent_total if d.should_notify else agent_metrics.agent_notifications_skipped_total).inc(); _log("agent.notification.sent" if d.should_notify else "agent.notification.skipped", reason=d.reason); return state
    @node("finalize")
    def finalize(state):
        tasks = state.get("open_tasks", []); blockers=[t for t in tasks if t.get("status")=="blocked" or t.get("blocker")]
        lines=["MemOS-Q Agent Briefing", "", f"What changed: {len(state.get('recent_conversations', []))} recent conversation updates reviewed.", f"Open goals/tasks: {len(tasks)}", f"Blockers: {len(blockers)}"]
        if tasks: lines.append("Recommended next action: " + (tasks[0].get("next_action") or f"Review task: {tasks[0].get('title')}"))
        if state.get("system_health"): lines.append("System issues: " + ("None" if not state.get("errors") else "; ".join(state["errors"][:3])))
        lines.append(f"What the agent did: ran {state.get('metadata',{}).get('tool_calls',0)} tool checks.")
        lines.append("Needs user input: " + ("Yes" if blockers else "No immediate input required."))
        state["final_briefing"] = truncate_telegram_message("\n".join(dict.fromkeys(lines)), limit=1500)
        return state
    try:
        from langgraph.graph import END, StateGraph
        from .state import AgentState
        g=StateGraph(AgentState)
        for name, fn in [("load_context",load_context),("retrieve_memory",retrieve_memory),("load_tasks",load_tasks),("check_system_health",health),("planner",planner),("tool_executor",tools_node),("reflect",reflect),("update_memory_and_tasks",update),("notification_decision",notify),("finalize",finalize)]: g.add_node(name, fn)
        g.set_entry_point("load_context"); g.add_edge("load_context","retrieve_memory"); g.add_edge("retrieve_memory","load_tasks"); g.add_edge("load_tasks","check_system_health"); g.add_edge("check_system_health","planner")
        g.add_conditional_edges("planner", route_after_planner, {"tool_executor":"tool_executor","finalize":"finalize"}); g.add_edge("tool_executor","reflect"); g.add_conditional_edges("reflect", route_after_reflection, {"planner":"planner","update_memory_and_tasks":"update_memory_and_tasks"}); g.add_edge("update_memory_and_tasks","notification_decision"); g.add_edge("notification_decision","finalize"); g.add_edge("finalize", END); return g.compile()
    except Exception:
        return SimpleCompiledGraph([load_context, retrieve_memory, load_tasks, health, planner, tools_node, reflect, update, notify, finalize])

def run_agentic(store: Any, qwen_client: Any, telegram_client: Any = None, *, user_id: str = "default", trigger: str = "manual_api_run", input_text: str | None = None, send: bool = False, force_send: bool = False) -> dict[str, Any]:
    run_id=str(uuid4()); start=time.perf_counter(); agent_metrics.agent_runs_total.labels(trigger=trigger).inc(); _log("agent.run.started", run_id=run_id)
    state={"user_id": user_id, "trigger": trigger, "input_text": input_text, "recent_conversations": [], "relevant_memories": [], "open_tasks": [], "system_health": {}, "plan": [], "current_step_index": 0, "observations": [], "task_updates": [], "final_briefing": None, "should_notify": False, "notification_reason": None, "errors": [], "metadata": {"run_id": run_id, "send": send, "force_send": force_send, "tool_calls": 0, "planning_rounds": 0, "replans": 0}}
    try: result=build_agent_graph(store, qwen_client, telegram_client).invoke(state); status="completed"
    except Exception as exc: result=state; result["errors"].append(str(exc)); status="failed"
    result["metadata"]["run_id"]=run_id; agent_metrics.agent_run_duration_seconds.observe(time.perf_counter()-start); _log("agent.run.completed", run_id=run_id, status=status)
    if hasattr(store, "add_agent_run"):
        store.add_agent_run(AgentRun(user_id=user_id, trigger=trigger, status=status, plan_json=result.get("plan", []), observations_json=result.get("observations", []), final_briefing=result.get("final_briefing"), should_notify=result.get("should_notify", False), notification_reason=result.get("notification_reason"), errors_json=result.get("errors", [])))
    return result

def run_agentic_daily_briefing(user_id: str, store: Any, qwen_client: Any, telegram_client: Any = None, *, force_send: bool = False) -> dict[str, Any]:
    return run_agentic(store, qwen_client, telegram_client, user_id=user_id, trigger="telegram_daily_briefing", input_text="Create today's daily briefing", force_send=force_send, send=force_send)
