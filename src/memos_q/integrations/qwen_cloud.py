"""Live QwenCloud model integrations.

The client uses DashScope's OpenAI-compatible HTTP surface so deployments only
need ``QWEN_API_KEY`` plus model names in ``.env``. Qwen-Agent orchestration is
provided through ``build_qwen_agent`` when the optional ``qwen-agent`` package
is installed in the runtime image.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import json
from typing import Any

import requests

from memos_q.config import Settings, settings


@dataclass(slots=True)
class QwenMessage:
    """Chat message for Qwen-compatible requests."""

    role: str
    content: str | list[dict[str, Any]]


class QwenCloudClient:
    """HTTP client for Qwen chat, vision, batch, and Alibaba embeddings."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.session = requests.Session()

    def chat(
        self,
        messages: Sequence[QwenMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a response with Qwen3.5-Plus by default."""

        payload = self._chat_payload(messages, model=model, temperature=temperature)
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: Sequence[QwenMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream generated chat tokens from Qwen's OpenAI-compatible endpoint."""

        payload = {**self._chat_payload(messages, model=model, temperature=temperature), "stream": True}
        for event in self._post_stream("/chat/completions", payload):
            if event == "[DONE]":
                break
            try:
                data = json.loads(event)
            except ValueError:
                continue
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    def _chat_payload(
        self,
        messages: Sequence[QwenMessage | dict[str, Any]],
        *,
        model: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "model": model or self.config.qwen_reasoning_model,
            "messages": [serialize_message(message) for message in messages],
            "temperature": temperature,
        }

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """Embed text through Alibaba Cloud Model Studio's OpenAI-compatible endpoint.

        DashScope documents ``POST /embeddings`` for ``text-embedding-v4`` and
        allows dimensions including 64 through 2,048. The method keeps input
        ordering and returns one vector per input string.
        """

        if not texts:
            return []
        payload = {
            "model": model or self.config.qwen_embedding_model,
            "input": list(texts),
            "dimensions": dimensions or self.config.qwen_embedding_dimensions,
            "encoding_format": "float",
        }
        data = self._post("/embeddings", payload)
        vectors_by_index = {item["index"]: item["embedding"] for item in data["data"]}
        return [list(map(float, vectors_by_index[index])) for index in range(len(texts))]

    def embed_text(self, text: str, *, model: str | None = None, dimensions: int | None = None) -> list[float]:
        """Embed one text string through Alibaba Cloud Model Studio."""

        return self.embed_texts([text], model=model, dimensions=dimensions)[0]

    def flash_classify(self, prompt: str, *, labels: Iterable[str]) -> str:
        """Classify or route text with the configured Qwen3.5-Flash model."""

        label_text = ", ".join(labels)
        return self.chat(
            [
                QwenMessage("system", f"Return exactly one label from: {label_text}."),
                QwenMessage("user", prompt),
            ],
            model=self.config.qwen_flash_model,
            temperature=0,
        ).strip()

    def vision_extract(self, *, image_url: str, prompt: str) -> str:
        """Extract memory-worthy information from an image or document URL."""

        return self.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            model=self.config.qwen_vl_model,
            temperature=0.1,
        )

    def create_batch(self, input_file_id: str, *, endpoint: str = "/v1/chat/completions") -> dict[str, Any]:
        """Create an offline Qwen batch job for compaction or cleanup."""

        return self._post(
            "/batches",
            {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": "24h",
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request(path, payload, stream=False)
        return response.json()

    def _post_stream(self, path: str, payload: dict[str, Any]) -> Iterator[str]:
        with self._request(path, payload, stream=True) as response:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                yield line.removeprefix("data:").strip()

    def _request(self, path: str, payload: dict[str, Any], *, stream: bool) -> requests.Response:
        if not self.config.qwen_api_key:
            raise RuntimeError("QWEN_API_KEY is required for live QwenCloud calls")
        response = self.session.post(
            self.config.qwen_base_url.rstrip("/") + path,
            headers={
                "Authorization": f"Bearer {self.config.qwen_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
            stream=stream,
        )
        response.raise_for_status()
        return response


def serialize_message(message: QwenMessage | dict[str, Any]) -> dict[str, Any]:
    """Serialize message dataclasses and pass dict messages through."""

    if isinstance(message, QwenMessage):
        return {"role": message.role, "content": message.content}
    return dict(message)


def build_qwen_agent(config: Settings = settings) -> Any:
    """Build a Qwen-Agent Assistant wired to the configured Qwen model."""

    from qwen_agent.agents import Assistant

    llm_cfg = {
        "model": config.qwen_reasoning_model,
        "model_server": config.qwen_base_url,
        "api_key": config.qwen_api_key,
    }
    return Assistant(
        llm=llm_cfg,
        system_message=(
            "You are MemOS-Q, an explainable memory operating system for AI agents. "
            "Use memory provenance, confidence, and auditability in every response."
        ),
    )
