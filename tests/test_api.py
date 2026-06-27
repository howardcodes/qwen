import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memos_q.api import app, memory_os
from memos_q.store import InMemoryStore
from memos_q.models import MemoryStatus


class TestEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        if "Rust" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def setup_function():
    memory_os.store = InMemoryStore()
    memory_os.embedding_provider = TestEmbeddingProvider()


def test_memories_requires_auth_and_uses_header_user():
    client = TestClient(app)
    response = client.post("/memories", json={"content": "User prefers concise explanations.", "user_id": "attacker"})
    assert response.status_code == 422

    response = client.post(
        "/memories",
        headers={"x-user-id": "real-user"},
        json={"content": "User prefers concise explanations.", "memory_type": "preference"},
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "real-user"
    assert response.json()["memory_type"] == "preference"


def test_validation_bounds_recall_limit_and_content_length():
    client = TestClient(app)
    response = client.post("/recall", headers={"x-user-id": "user-1"}, json={"query": "x", "limit": 21})
    assert response.status_code == 422

    response = client.post("/memories", headers={"x-user-id": "user-1"}, json={"content": ""})
    assert response.status_code == 422


def test_per_user_recall_is_isolated():
    client = TestClient(app)
    client.post("/memories", headers={"x-user-id": "user-a"}, json={"content": "User prefers Python."})
    client.post("/memories", headers={"x-user-id": "user-b"}, json={"content": "User prefers Rust."})

    response = client.post("/recall", headers={"x-user-id": "user-a"}, json={"query": "Rust Python", "limit": 10})

    assert response.status_code == 200
    contents = [item["memory"]["content"] for item in response.json()]
    assert "User prefers Python." in contents
    assert "User prefers Rust." not in contents


def test_integrations_status_reports_runtime_sections():
    client = TestClient(app)

    response = client.get("/integrations/status")

    assert response.status_code == 200
    payload = response.json()
    assert "qwen_agent_available" in payload["backend"]
    assert "qwen_agent_package_available" in payload["backend"]
    assert payload["jobs"]["celery_broker_url_configured"] is True
    assert payload["jobs"]["celery_result_backend_configured"] is True
    assert "langfuse_configured" in payload["monitoring"]
    assert payload["monitoring"]["prometheus_metrics_path"] == "/metrics"
    assert payload["models"]["reasoning_model"]



def test_agent_chat_uses_filtered_active_memory_context(monkeypatch):
    client = TestClient(app)
    memory_os.remember(
        user_id="demo-user",
        content="User's name is Mark",
        source_session="session-name",
        status=MemoryStatus.PENDING_REVIEW,
    )
    memory_os.remember(user_id="demo-user", content="User likes to play badminton", source_session="session-sport")
    captured = {}

    def fake_stream(messages, model=None):
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr("memos_q.api.qwen_client.chat_stream", fake_stream)
    monkeypatch.setattr("memos_q.api.enqueue_memory_evolution", lambda **kwargs: None)

    response = client.post("/agent/chat", headers={"x-user-id": "demo-user"}, json={"message": "What is my name?"})

    assert response.status_code == 200
    prompt = captured["messages"][1].content
    assert "User's name is Mark" not in prompt
    assert "pending_review" not in prompt
    assert "User likes to play badminton" in prompt


def test_agent_chat_formats_structured_profile_without_raw_profile_stream(monkeypatch):
    client = TestClient(app)
    memory_os.store.upsert_user_profile("profile-user", name="Mark", age=25)
    memory_os.store.add_memory_stream_entry(
        __import__("memos_q.models", fromlist=["MemoryStreamEntry"]).MemoryStreamEntry(
            user_id="profile-user",
            content="User's name is Mark.",
            kind="profile",
            importance_score=8,
            metadata={"source_session": "s1", "stream_kind": "profile"},
        )
    )
    captured = {}

    def fake_stream(messages, model=None, max_tokens=None):
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr("memos_q.api.qwen_client.chat_stream", fake_stream)
    monkeypatch.setattr("memos_q.api.enqueue_memory_evolution", lambda **kwargs: None)

    response = client.post("/agent/chat", headers={"x-user-id": "profile-user"}, json={"message": "What is my name?"})

    assert response.status_code == 200
    prompt = captured["messages"][1].content
    assert "User profile:" in prompt
    assert "- Name: Mark" in prompt
    assert "- Age: 25" in prompt
    assert "User's name is Mark." not in prompt
