"""Durable production integrations for PostgreSQL, pgvector, Redis, and S3."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from memos_q.config import Settings, settings
from memos_q.models import AuditEvent, Memory, MemoryEdge, MemoryStatus, MemoryType, RelationType, utc_now
from memos_q.store import memory_snapshot


class PostgresMemoryStore:
    """PostgreSQL + pgvector persistence adapter for MemOS-Q memories."""

    def __init__(self, config: Settings = settings, *, connect_retries: int = 30, retry_delay: float = 2.0) -> None:
        import psycopg

        self.config = config
        last_error: Exception | None = None
        for attempt in range(1, connect_retries + 1):
            try:
                self.connection = psycopg.connect(config.postgres_dsn)
                break
            except psycopg.OperationalError as error:
                last_error = error
                if attempt == connect_retries:
                    raise
                time.sleep(retry_delay)
        else:  # pragma: no cover - defensive; loop either breaks or raises
            raise RuntimeError("PostgreSQL connection was not initialized") from last_error

    def migrate(self) -> None:
        """Create required tables and pgvector extension if missing."""

        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.config.qwen_embedding_dimensions}),
                    confidence_score DOUBLE PRECISION NOT NULL,
                    importance_score DOUBLE PRECISION NOT NULL,
                    novelty_score DOUBLE PRECISION NOT NULL,
                    stability_score DOUBLE PRECISION NOT NULL,
                    confidence_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source_session TEXT NOT NULL,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    last_recalled_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id TEXT PRIMARY KEY,
                    source_memory TEXT NOT NULL REFERENCES memories(id),
                    target_memory TEXT NOT NULL REFERENCES memories(id),
                    relation_type TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    previous_value JSONB,
                    new_value JSONB,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops)")
        self.connection.commit()

    def add_memory(self, memory: Memory, *, actor: str = "memory-agent") -> Memory:
        """Persist a memory record."""

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memories (
                    id, user_id, memory_type, content, confidence_score,
                    importance_score, novelty_score, stability_score, confidence_reasons,
                    status, version, source_session, tags, metadata, created_at, updated_at,
                    last_recalled_at, expires_at, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.memory_type.value,
                    memory.content,
                    memory.confidence_score,
                    memory.importance_score,
                    memory.novelty_score,
                    memory.stability_score,
                    json.dumps(memory.confidence_reasons),
                    memory.status.value,
                    memory.version,
                    memory.source_session,
                    json.dumps(sorted(memory.tags)),
                    json.dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                    memory.last_recalled_at,
                    memory.expires_at,
                    memory.embedding,
                ),
            )
            self._record_audit(cursor, "remember", actor, memory.id, None, memory_snapshot(memory))
        self.connection.commit()
        return memory

    def get_memory(self, memory_id: str) -> Memory:
        """Load one memory by id."""

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, memory_type, content, confidence_score,
                       importance_score, novelty_score, stability_score, confidence_reasons, status,
                       version, source_session, tags, metadata, created_at,
                       updated_at, last_recalled_at, expires_at, embedding
                FROM memories
                WHERE id = %s
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(memory_id)
        return memory_from_row(row)

    def update_memory(self, memory_id: str, *, actor: str = "memory-agent", **changes: object) -> Memory:
        """Update a persisted memory and append an audit record."""

        memory = self.get_memory(memory_id)
        previous = memory_snapshot(memory)
        for key, value in changes.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        memory.version += 1
        memory.updated_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE memories
                SET memory_type = %s, content = %s, confidence_score = %s,
                    importance_score = %s, novelty_score = %s, stability_score = %s,
                    confidence_reasons = %s, status = %s, version = %s, tags = %s,
                    metadata = %s, updated_at = %s, last_recalled_at = %s,
                    expires_at = %s, embedding = %s
                WHERE id = %s
                """,
                (
                    memory.memory_type.value,
                    memory.content,
                    memory.confidence_score,
                    memory.importance_score,
                    memory.novelty_score,
                    memory.stability_score,
                    json.dumps(memory.confidence_reasons),
                    memory.status.value,
                    memory.version,
                    json.dumps(sorted(memory.tags)),
                    json.dumps(memory.metadata),
                    memory.updated_at,
                    memory.last_recalled_at,
                    memory.expires_at,
                    memory.embedding,
                    memory.id,
                ),
            )
            self._record_audit(cursor, "update", actor, memory.id, previous, memory_snapshot(memory))
        self.connection.commit()
        return memory

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """Persist a graph relationship between memories."""

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory_edges (id, source_memory, target_memory, relation_type, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (edge.id, edge.source_memory, edge.target_memory, edge.relation_type.value, edge.created_at),
            )
            self._record_audit(
                cursor,
                "edge_create",
                "memory-graph",
                edge.source_memory,
                None,
                {
                    "source_memory": edge.source_memory,
                    "target_memory": edge.target_memory,
                    "relation_type": edge.relation_type.value,
                },
            )
        self.connection.commit()
        return edge

    def edges_for(self, memory_id: str, relation_type: RelationType | None = None) -> list[MemoryEdge]:
        """Load graph edges touching one memory."""

        relation_clause = "" if relation_type is None else "AND relation_type = %s"
        params: tuple[object, ...] = (memory_id, memory_id) if relation_type is None else (memory_id, memory_id, relation_type.value)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, source_memory, target_memory, relation_type, created_at
                FROM memory_edges
                WHERE (source_memory = %s OR target_memory = %s) {relation_clause}
                ORDER BY created_at
                """,
                params,
            )
            return [
                MemoryEdge(
                    id=row[0],
                    source_memory=row[1],
                    target_memory=row[2],
                    relation_type=RelationType(row[3]),
                    created_at=row[4],
                )
                for row in cursor.fetchall()
            ]

    def list_user_ids(self) -> list[str]:
        """Load ids for users that have memory records."""

        with self.connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT user_id FROM memories ORDER BY user_id")
            return [row[0] for row in cursor.fetchall()]

    def audit_log(self, memory_id: str | None = None) -> list[AuditEvent]:
        """Load audit log entries."""

        clause = "" if memory_id is None else "WHERE memory_id = %s"
        params: tuple[object, ...] = () if memory_id is None else (memory_id,)
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT action, actor, memory_id, previous_value, new_value, timestamp
                FROM audit_log
                {clause}
                ORDER BY timestamp
                """,
                params,
            )
            return [
                AuditEvent(
                    action=row[0],
                    actor=row[1],
                    memory_id=row[2],
                    previous_value=row[3],
                    new_value=row[4],
                    timestamp=row[5],
                )
                for row in cursor.fetchall()
            ]

    def list_memories(self, user_id: str, *, include_inactive: bool = False) -> list[Memory]:
        """Load memories for a user."""

        status_clause = "" if include_inactive else "AND status = 'active'"
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, memory_type, content, confidence_score,
                       importance_score, novelty_score, stability_score, confidence_reasons, status,
                       version, source_session, tags, metadata, created_at,
                       updated_at, last_recalled_at, expires_at, embedding
                FROM memories
                WHERE user_id = %s {status_clause}
                ORDER BY created_at
                """,
                (user_id,),
            )
            return [memory_from_row(row) for row in cursor.fetchall()]

    def vector_search(self, user_id: str, query_embedding: list[float], *, limit: int = 20) -> list[Memory]:
        """Load nearest active memories using pgvector cosine distance."""

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, memory_type, content, confidence_score,
                       importance_score, novelty_score, stability_score, confidence_reasons, status,
                       version, source_session, tags, metadata, created_at,
                       updated_at, last_recalled_at, expires_at, embedding
                FROM memories
                WHERE user_id = %s AND status = 'active' AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (user_id, query_embedding, limit),
            )
            return [memory_from_row(row) for row in cursor.fetchall()]

    def _record_audit(
        self,
        cursor: Any,
        action: str,
        actor: str,
        memory_id: str,
        previous_value: dict[str, object] | None,
        new_value: dict[str, object] | None,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO audit_log (action, actor, memory_id, previous_value, new_value)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (action, actor, memory_id, json.dumps(previous_value), json.dumps(new_value)),
        )


class RedisMemoryCache:
    """Redis cache for hot recall results and session state."""

    def __init__(self, config: Settings = settings) -> None:
        import redis

        self.client = redis.Redis.from_url(config.redis_url, decode_responses=True)

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int = 300) -> None:
        self.client.setex(key, ttl_seconds, json.dumps(value))

    def get_json(self, key: str) -> dict[str, Any] | None:
        raw = self.client.get(key)
        return json.loads(raw) if raw else None


class S3ObjectStore:
    """S3-compatible object storage for PDFs, screenshots, and batch files."""

    def __init__(self, config: Settings = settings) -> None:
        import boto3

        self.config = config
        self.client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            aws_access_key_id=config.s3_access_key_id,
            aws_secret_access_key=config.s3_secret_access_key,
            region_name=config.s3_region,
        )

    def ensure_bucket(self) -> None:
        buckets = self.client.list_buckets().get("Buckets", [])
        if not any(bucket["Name"] == self.config.s3_bucket for bucket in buckets):
            self.client.create_bucket(Bucket=self.config.s3_bucket)

    def put_text(self, key: str, content: str, *, content_type: str = "text/plain") -> str:
        self.client.put_object(
            Bucket=self.config.s3_bucket,
            Key=key,
            Body=content.encode(),
            ContentType=content_type,
        )
        return f"{self.config.s3_endpoint_url.rstrip('/')}/{self.config.s3_bucket}/{key}"


def memory_from_row(row: Iterable[Any]) -> Memory:
    """Convert a PostgreSQL row into a Memory object."""

    (
        memory_id,
        user_id,
        memory_type,
        content,
        confidence_score,
        importance_score,
        novelty_score,
        stability_score,
        confidence_reasons,
        status,
        version,
        source_session,
        tags,
        metadata,
        created_at,
        updated_at,
        last_recalled_at,
        expires_at,
        embedding,
    ) = row
    if isinstance(tags, str):
        tags = json.loads(tags)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(confidence_reasons, str):
        confidence_reasons = json.loads(confidence_reasons)
    return Memory(
        id=memory_id,
        user_id=user_id,
        memory_type=MemoryType(memory_type),
        content=content,
        confidence_score=confidence_score,
        importance_score=importance_score,
        novelty_score=novelty_score,
        stability_score=stability_score,
        status=MemoryStatus(status),
        version=version,
        source_session=source_session,
        tags=set(tags or []),
        metadata=dict(metadata or {}),
        confidence_reasons=list(confidence_reasons or []),
        embedding=list(embedding or []),
        created_at=created_at,
        updated_at=updated_at,
        last_recalled_at=last_recalled_at,
        expires_at=expires_at,
    )
