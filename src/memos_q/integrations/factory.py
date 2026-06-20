"""Factories that choose local or live infrastructure based on settings."""

from __future__ import annotations

from memos_q.config import Settings, settings
from memos_q.integrations.durable import AliCloudMemoryStore, PostgresMemoryStore
from memos_q.store import InMemoryStore


def build_memory_store(config: Settings = settings) -> InMemoryStore | PostgresMemoryStore:
    """Return the configured memory store.

    ``MEMOS_STORE=alicloud`` is production mode: Postgres on ECS stores memory
    records and Pinecone performs vector recall.
    """

    if config.memos_store.lower() in {"alicloud", "ecs"}:
        store = AliCloudMemoryStore(config)
        store.migrate()
        return store
    if config.memos_store.lower() == "postgres":
        store = PostgresMemoryStore(config)
        store.migrate()
        return store
    return InMemoryStore()
