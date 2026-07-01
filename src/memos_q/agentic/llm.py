"""QwenCloud helpers for LangChain/LangGraph agent nodes."""
from __future__ import annotations
import json
from typing import Any
from memos_q.config import settings
from memos_q.integrations.qwen_cloud import QwenMessage

class QwenChatHelper:
    def __init__(self, client: Any) -> None:
        self.client = client
    def chat(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        return self.client.chat([QwenMessage("system", system), QwenMessage("user", user)], model=settings.qwen_flash_model, temperature=0, max_tokens=max_tokens or settings.qwen_summary_max_tokens)
    def json(self, system: str, user: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = self.chat(system, user)
            start, end = raw.find("{"), raw.rfind("}")
            if not raw.strip():
                raise ValueError("qwen returned empty response")
            if start >= 0 and end >= start:
                raw = raw[start:end+1]
            data = json.loads(raw)
            return data if isinstance(data, dict) else fallback
        except Exception as exc:
            out = dict(fallback); out.setdefault("errors", []).append(str(exc)); return out
