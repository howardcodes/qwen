"""Factories that choose local or live infrastructure based on settings."""

from __future__ import annotations

from memos_q.config import Settings, settings
from memos_q.integrations.durable import PostgresMemoryStore
from memos_q.store import InMemoryStore


def build_memory_store(config: Settings = settings) -> InMemoryStore | PostgresMemoryStore:
    """Return PostgreSQL storage when MEMOS_STORE=postgres, otherwise memory."""

    if config.memos_store.lower() == "postgres":
        store = PostgresMemoryStore(config)
        store.migrate()
        return store
    return InMemoryStore()
