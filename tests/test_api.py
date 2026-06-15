import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memos_q.api import app, memory_os
from memos_q.store import InMemoryStore


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
