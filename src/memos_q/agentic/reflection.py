"""Reflection / critique node for agentic runs."""
from __future__ import annotations
from typing import Any

def reflect_on_run(state: dict[str, Any], qwen_client: Any | None = None) -> dict[str, Any]:
    failures = [o for o in state.get("observations", []) if not o.get("ok", True)]
    needs = bool(failures) and state.get("metadata", {}).get("replans", 0) < 1
    score = 0.8 if state.get("observations") and not failures else 0.45
    result = {"needs_replan": needs, "reason": "tool failures require one safer replan" if needs else "sufficient evidence collected", "missing_info": [f.get("error", "tool failure") for f in failures], "final_quality_score": score}
    state.setdefault("metadata", {})["reflection"] = result
    state["metadata"]["needs_replan"] = needs
    if needs:
        state["metadata"]["replans"] = state["metadata"].get("replans", 0) + 1
        state["plan"] = []
    return state
