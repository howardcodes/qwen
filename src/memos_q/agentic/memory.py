"""Memory retrieval and safe update logic for agentic runs."""
from __future__ import annotations
from datetime import timedelta
from typing import Any
from memos_q.models import Memory, MemoryStatus, MemoryType, utc_now

def _mem(m: Any) -> dict[str, Any]:
    return {"id": m.id, "content": m.content, "memory_type": getattr(m.memory_type, "value", str(m.memory_type)), "status": getattr(m.status, "value", str(m.status)), "updated_at": m.updated_at.isoformat(), "importance_score": m.importance_score}

def _task(t: Any) -> dict[str, Any]:
    return {"id": t.id, "title": t.title, "status": getattr(t.status, "value", str(t.status)), "blocker_type": t.blocker_type, "blocker": t.blocker, "next_action": t.next_action, "evidence": list(t.evidence), "confidence": t.confidence, "updated_at": t.updated_at.isoformat(), "metadata": dict(t.metadata)}

def load_agent_context(user_id: str, store: Any) -> dict[str, Any]:
    since = utc_now() - timedelta(hours=24)
    return {"recent_conversations": store.recent_conversation_activity(user_id, since=since, limit=80) if hasattr(store, "recent_conversation_activity") else [], "open_tasks": [_task(t) for t in store.list_task_records(user_id)[:20]] if hasattr(store, "list_task_records") else []}

def retrieve_relevant_memories(user_id: str, query: str | None, store: Any) -> list[dict[str, Any]]:
    memories = store.list_memories(user_id) if hasattr(store, "list_memories") else []
    terms = {w.lower() for w in (query or "").split() if len(w) > 3}
    scored = sorted(memories, key=lambda m: (sum(1 for w in terms if w in m.content.lower()), m.importance_score, m.updated_at), reverse=True)
    return [_mem(m) for m in scored[:20] if m.status == MemoryStatus.ACTIVE]

def propose_memory_updates(state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates=[]
    for obs in state.get("observations", []):
        data=obs.get("data", {}) if isinstance(obs, dict) else {}
        content=data.get("memory_candidate") or data.get("content")
        if content and len(str(content).split()) >= 5:
            candidates.append({"content": str(content)[:500], "evidence": obs, "confidence": 0.7})
    return candidates[:3]

def persist_memory_updates(state: dict[str, Any], store: Any) -> list[dict[str, Any]]:
    saved=[]
    if not hasattr(store, "add_memory"):
        return saved
    for c in propose_memory_updates(state):
        mem = Memory(user_id=state["user_id"], content=c["content"], memory_type=MemoryType.CONVERSATION_SUMMARY, source_session="agentic-run", status=MemoryStatus.PENDING_REVIEW, confidence_score=c["confidence"], metadata={"source": "agentic_candidate", "evidence": c["evidence"]})
        saved.append(_mem(store.add_memory(mem, actor="agentic-memory")))
    return saved
