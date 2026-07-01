"""Prometheus metrics for LangGraph agentic runs."""
from prometheus_client import Counter, Histogram

agent_runs_total = Counter("agent_runs_total", "Total agentic runs", ["trigger"])
agent_tool_calls_total = Counter("agent_tool_calls_total", "Total agent tool calls")
agent_tool_failures_total = Counter("agent_tool_failures_total", "Total failed agent tool calls")
agent_replans_total = Counter("agent_replans_total", "Total agent replans")
agent_notifications_sent_total = Counter("agent_notifications_sent_total", "Agent notifications approved or sent")
agent_notifications_skipped_total = Counter("agent_notifications_skipped_total", "Agent notifications skipped")
agent_run_duration_seconds = Histogram("agent_run_duration_seconds", "Agent run duration in seconds")
