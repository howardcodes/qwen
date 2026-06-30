"""Durable production integrations for PostgreSQL, Pinecone, Redis, and MinIO/S3."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from memos_q.config import Settings, settings
from memos_q.models import AuditEvent, ChatTurn, DailySummary, DailySummarySettings, Memory, UserProfile, TaskRecord, TaskRecordStatus, MemoryStreamEntry, MemoryStreamKind, MemoryConflict, MemoryConflictStatus, MemoryEdge, MemoryStatus, MemoryType, RelationType, SessionState, utc_now
from memos_q.store import memory_snapshot, memory_stream_snapshot, profile_snapshot, session_state_snapshot, task_record_snapshot


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
            use_pgvector = self.config.memos_store.lower() == "postgres"
            if use_pgvector:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            embedding_type = f"vector({self.config.qwen_embedding_dimensions})" if use_pgvector else "JSONB"
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding {embedding_type},
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
                    approved_at TIMESTAMPTZ,
                    last_seen_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ,
                    sensitivity TEXT NOT NULL DEFAULT 'low',
                    conflicting_memory_id TEXT,
                    conflict_reason TEXT,
                    last_confirmed_at TIMESTAMPTZ
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
                CREATE TABLE IF NOT EXISTS memory_conflicts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    existing_memory_id TEXT NOT NULL REFERENCES memories(id),
                    candidate_memory_id TEXT NOT NULL REFERENCES memories(id),
                    conflict_type TEXT NOT NULL,
                    existing_content TEXT NOT NULL,
                    candidate_content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    resolved_at TIMESTAMPTZ,
                    resolution TEXT
                );
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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_stream (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    importance_score INTEGER NOT NULL,
                    last_accessed_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    decay_rate DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    occupation TEXT,
                    timezone TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_summary_settings (
                    user_id TEXT PRIMARY KEY,
                    enabled BOOLEAN NOT NULL DEFAULT false,
                    summary_time TEXT NOT NULL DEFAULT '09:00',
                    timezone TEXT NOT NULL DEFAULT 'Asia/Singapore',
                    telegram_chat_id TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    sent_to_telegram BOOLEAN NOT NULL DEFAULT false,
                    sent_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL,
                    error_message TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS task_records (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    blocker_type TEXT NOT NULL DEFAULT 'none',
                    blocker TEXT,
                    next_action TEXT,
                    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.75,
                    source TEXT NOT NULL DEFAULT 'daily_briefing',
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (user_id, title)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS session_state (
                    user_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    current_topic TEXT,
                    active_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
                    open_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    user_goal TEXT,
                    constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, conversation_id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_task_records_user_status_updated ON task_records(user_id, status, updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_turns_user_conversation_created ON conversation_turns(user_id, conversation_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_state_user_conversation ON session_state(user_id, conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_stream_user_created ON memory_stream(user_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_stream_user_status ON memory_stream(user_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_summaries_user_created ON daily_summaries(user_id, created_at DESC)")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS sensitivity TEXT NOT NULL DEFAULT 'low'")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS conflicting_memory_id TEXT")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS conflict_reason TEXT")
            cursor.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_confirmed_at TIMESTAMPTZ")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_conflicts_user_status ON memory_conflicts(user_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_updated ON memories(user_id, updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags)")
            if use_pgvector:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops)")
        self.connection.commit()


    def append_conversation_turn(self, user_id: str, conversation_id: str, turn: ChatTurn, *, max_turns: int = 50) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversation_turns (user_id, conversation_id, role, content) VALUES (%s, %s, %s, %s)",
                (user_id, conversation_id, turn.role, turn.content),
            )
            cursor.execute(
                """
                DELETE FROM conversation_turns
                WHERE id IN (
                    SELECT id FROM conversation_turns
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY created_at DESC, id DESC OFFSET %s
                )
                """,
                (user_id, conversation_id, max_turns),
            )
            self._record_audit(cursor, "conversation_turn_append", "conversation_turns", conversation_id, None, {"role": turn.role, "content": turn.content})
        self.connection.commit()

    def recent_conversation_turns(self, user_id: str, conversation_id: str, *, limit: int = 10) -> list[ChatTurn]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at, id FROM conversation_turns
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY created_at DESC, id DESC LIMIT %s
                ) recent ORDER BY created_at, id
                """,
                (user_id, conversation_id, limit),
            )
            return [ChatTurn(role=row[0], content=row[1]) for row in cursor.fetchall()]

    def recent_conversation_activity(self, user_id: str, *, since, limit: int = 80) -> list[dict[str, str]]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content FROM conversation_turns
                WHERE user_id = %s AND created_at >= %s
                ORDER BY created_at DESC, id DESC LIMIT %s
                """,
                (user_id, since, limit),
            )
            rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    def get_daily_summary_settings(self, user_id: str) -> DailySummarySettings:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT user_id, enabled, summary_time, timezone, telegram_chat_id, updated_at FROM daily_summary_settings WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
        if not row:
            return DailySummarySettings(user_id=user_id)
        return DailySummarySettings(user_id=row[0], enabled=row[1], summary_time=row[2], timezone=row[3], telegram_chat_id=row[4], updated_at=row[5])

    def update_daily_summary_settings(self, user_id: str, **changes: object) -> DailySummarySettings:
        current = self.get_daily_summary_settings(user_id)
        for key, value in changes.items():
            if hasattr(current, key):
                setattr(current, key, value)
        current.updated_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO daily_summary_settings (user_id, enabled, summary_time, timezone, telegram_chat_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET enabled = EXCLUDED.enabled, summary_time = EXCLUDED.summary_time,
                    timezone = EXCLUDED.timezone, telegram_chat_id = EXCLUDED.telegram_chat_id, updated_at = EXCLUDED.updated_at
                """,
                (current.user_id, current.enabled, current.summary_time, current.timezone, current.telegram_chat_id, current.updated_at),
            )
            self._record_audit(cursor, "daily_summary_settings_update", "user", user_id, None, {"enabled": current.enabled, "summary_time": current.summary_time, "timezone": current.timezone, "telegram_chat_id": current.telegram_chat_id})
        self.connection.commit()
        return current

    def add_daily_summary(self, summary: DailySummary) -> DailySummary:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO daily_summaries (id, user_id, summary_text, sent_to_telegram, sent_at, created_at, error_message) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (summary.id, summary.user_id, summary.summary_text, summary.sent_to_telegram, summary.sent_at, summary.created_at, summary.error_message),
            )
            self._record_audit(cursor, "daily_summary_create", "daily-summary-agent", summary.user_id, None, {"summary_id": summary.id, "sent_to_telegram": summary.sent_to_telegram, "error_message": summary.error_message})
        self.connection.commit()
        return summary

    def upsert_task_record(self, task: TaskRecord, *, actor: str = "daily-briefing-agent") -> TaskRecord:
        task.updated_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO task_records (id, user_id, title, status, blocker_type, blocker, next_action, evidence, confidence, source, metadata, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, title) DO UPDATE SET
                    status = EXCLUDED.status, blocker_type = EXCLUDED.blocker_type, blocker = EXCLUDED.blocker,
                    next_action = EXCLUDED.next_action, evidence = EXCLUDED.evidence, confidence = EXCLUDED.confidence,
                    source = EXCLUDED.source, metadata = EXCLUDED.metadata, updated_at = EXCLUDED.updated_at
                RETURNING id, created_at
                """,
                (task.id, task.user_id, task.title, task.status.value, task.blocker_type, task.blocker, task.next_action, json.dumps(task.evidence), task.confidence, task.source, json.dumps(task.metadata), task.created_at, task.updated_at),
            )
            row = cursor.fetchone()
            if row:
                task.id, task.created_at = row[0], row[1]
            self._record_audit(cursor, "task_record_upsert", actor, task.id, None, task_record_snapshot(task))
        self.connection.commit()
        return task

    def list_task_records(self, user_id: str, *, include_closed: bool = False) -> list[TaskRecord]:
        status_clause = "" if include_closed else "AND status NOT IN ('done', 'dropped')"
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, title, status, blocker_type, blocker, next_action, evidence, confidence, source, metadata, created_at, updated_at
                FROM task_records WHERE user_id = %s {status_clause} ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            return [task_record_from_row(row) for row in cursor.fetchall()]

    def get_session_state(self, user_id: str, conversation_id: str) -> SessionState:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT current_topic, active_entities, open_questions, user_goal, constraints, updated_at
                FROM session_state WHERE user_id = %s AND conversation_id = %s
                """,
                (user_id, conversation_id),
            )
            row = cursor.fetchone()
        if not row:
            return SessionState()
        return SessionState(current_topic=row[0], active_entities=list(row[1] or []), open_questions=list(row[2] or []), user_goal=row[3], constraints=list(row[4] or []), updated_at=row[5])

    def update_session_state(self, user_id: str, conversation_id: str, state: SessionState) -> SessionState:
        state.updated_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO session_state (user_id, conversation_id, current_topic, active_entities, open_questions, user_goal, constraints, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, conversation_id) DO UPDATE SET
                    current_topic = EXCLUDED.current_topic, active_entities = EXCLUDED.active_entities,
                    open_questions = EXCLUDED.open_questions, user_goal = EXCLUDED.user_goal,
                    constraints = EXCLUDED.constraints, updated_at = EXCLUDED.updated_at
                """,
                (user_id, conversation_id, state.current_topic, json.dumps(state.active_entities), json.dumps(state.open_questions), state.user_goal, json.dumps(state.constraints), state.updated_at),
            )
            self._record_audit(cursor, "session_state_update", "session_state", conversation_id, None, session_state_snapshot(state))
        self.connection.commit()
        return state

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT user_id, name, age, occupation, timezone, updated_at FROM user_profiles WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
        return UserProfile(user_id=row[0], name=row[1], age=row[2], occupation=row[3], timezone=row[4], updated_at=row[5]) if row else None

    def upsert_user_profile(self, user_id: str, *, actor: str = "profile-agent", **changes: object) -> UserProfile:
        previous = self.get_user_profile(user_id)
        current = previous or UserProfile(user_id=user_id)
        for key, value in changes.items():
            if hasattr(current, key) and value not in (None, ""):
                setattr(current, key, value)
        current.updated_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_profiles (user_id, name, age, occupation, timezone, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, user_profiles.name),
                    age = COALESCE(EXCLUDED.age, user_profiles.age),
                    occupation = COALESCE(EXCLUDED.occupation, user_profiles.occupation),
                    timezone = COALESCE(EXCLUDED.timezone, user_profiles.timezone),
                    updated_at = EXCLUDED.updated_at
                """,
                (current.user_id, current.name, current.age, current.occupation, current.timezone, current.updated_at),
            )
            self._record_audit(cursor, "profile_upsert", actor, user_id, profile_snapshot(previous) if previous else None, profile_snapshot(current))
        self.connection.commit()
        return current

    def add_memory_stream_entry(self, entry: MemoryStreamEntry, *, actor: str = "memory-stream") -> MemoryStreamEntry:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory_stream (id, user_id, content, kind, importance_score, last_accessed_at, created_at, decay_rate, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (entry.id, entry.user_id, entry.content, entry.kind.value, entry.importance_score, entry.last_accessed_at, entry.created_at, entry.decay_rate, entry.status.value, json.dumps(entry.metadata)),
            )
            self._record_audit(cursor, "stream_append", actor, entry.id, None, memory_stream_snapshot(entry))
        self.connection.commit()
        return entry

    def list_memory_stream(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryStreamEntry]:
        status_clause = "" if include_inactive else "AND status = 'active'"
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, content, kind, importance_score, last_accessed_at, created_at, decay_rate, status, metadata
                FROM memory_stream WHERE user_id = %s {status_clause} ORDER BY created_at
                """,
                (user_id,),
            )
            return [memory_stream_from_row(row) for row in cursor.fetchall()]

    def update_memory_stream_access(self, entry_ids: Iterable[str], *, actor: str = "recall") -> None:
        if not entry_ids:
            return
        with self.connection.cursor() as cursor:
            cursor.execute("UPDATE memory_stream SET last_accessed_at = now() WHERE id = ANY(%s)", (list(entry_ids),))
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
                    last_recalled_at, approved_at, last_seen_at, expires_at, embedding, sensitivity,
                    conflicting_memory_id, conflict_reason, last_confirmed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    memory.approved_at,
                    memory.last_seen_at,
                    memory.expires_at,
                    json.dumps(memory.embedding) if self.config.memos_store.lower() == "alicloud" else memory.embedding,
                    memory.sensitivity,
                    memory.conflicting_memory_id,
                    memory.conflict_reason,
                    memory.last_confirmed_at,
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
                       updated_at, last_recalled_at, approved_at, last_seen_at, expires_at, embedding, sensitivity, conflicting_memory_id, conflict_reason, last_confirmed_at
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
                    approved_at = %s, last_seen_at = %s, expires_at = %s, embedding = %s, sensitivity = %s, conflicting_memory_id = %s, conflict_reason = %s, last_confirmed_at = %s
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
                    memory.approved_at,
                    memory.last_seen_at,
                    memory.expires_at,
                    json.dumps(memory.embedding) if self.config.memos_store.lower() == "alicloud" else memory.embedding,
                    memory.sensitivity,
                    memory.conflicting_memory_id,
                    memory.conflict_reason,
                    memory.last_confirmed_at,
                    memory.id,
                ),
            )
            self._record_audit(cursor, "update", actor, memory.id, previous, memory_snapshot(memory))
        self.connection.commit()
        return memory


    def add_conflict(self, conflict: MemoryConflict) -> MemoryConflict:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory_conflicts (id, user_id, existing_memory_id, candidate_memory_id, conflict_type, existing_content, candidate_content, status, created_at, resolved_at, resolution)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (conflict.id, conflict.user_id, conflict.existing_memory_id, conflict.candidate_memory_id, conflict.conflict_type, conflict.existing_content, conflict.candidate_content, conflict.status.value, conflict.created_at, conflict.resolved_at, conflict.resolution),
            )
            self._record_audit(cursor, "conflict_create", "memory-conflict", conflict.candidate_memory_id, None, {"id": conflict.id, "status": conflict.status.value})
        self.connection.commit()
        return conflict

    def get_conflict(self, conflict_id: str) -> MemoryConflict:
        with self.connection.cursor() as cursor:
            cursor.execute("""SELECT id, user_id, existing_memory_id, candidate_memory_id, conflict_type, existing_content, candidate_content, status, created_at, resolved_at, resolution FROM memory_conflicts WHERE id = %s""", (conflict_id,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError(conflict_id)
        return conflict_from_row(row)

    def pending_conflict_for_user(self, user_id: str) -> MemoryConflict | None:
        with self.connection.cursor() as cursor:
            cursor.execute("""SELECT id, user_id, existing_memory_id, candidate_memory_id, conflict_type, existing_content, candidate_content, status, created_at, resolved_at, resolution FROM memory_conflicts WHERE user_id = %s AND status = 'pending' ORDER BY created_at DESC LIMIT 1""", (user_id,))
            row = cursor.fetchone()
        return conflict_from_row(row) if row else None

    def resolve_conflict(self, conflict_id: str, *, resolution: str, actor: str = "memory-agent") -> MemoryConflict:
        conflict = self.get_conflict(conflict_id)
        conflict.status = MemoryConflictStatus.RESOLVED
        conflict.resolution = resolution
        conflict.resolved_at = utc_now()
        with self.connection.cursor() as cursor:
            cursor.execute("""UPDATE memory_conflicts SET status = %s, resolved_at = %s, resolution = %s WHERE id = %s""", (conflict.status.value, conflict.resolved_at, conflict.resolution, conflict.id))
            self._record_audit(cursor, "conflict_resolve", actor, conflict.candidate_memory_id, None, {"id": conflict.id, "resolution": resolution})
        self.connection.commit()
        return conflict

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
            cursor.execute("SELECT user_id FROM memories UNION SELECT user_id FROM daily_summary_settings ORDER BY user_id")
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
                       updated_at, last_recalled_at, approved_at, last_seen_at, expires_at, embedding, sensitivity, conflicting_memory_id, conflict_reason, last_confirmed_at
                FROM memories
                WHERE user_id = %s {status_clause}
                ORDER BY created_at
                """,
                (user_id,),
            )
            return [memory_from_row(row) for row in cursor.fetchall()]

    def vector_search(self, user_id: str, query_embedding: list[float], *, limit: int = 20, include_inactive: bool = False) -> list[Memory]:
        """Load nearest active memories using pgvector cosine distance."""

        status_clause = "AND status != 'forgotten'" if include_inactive else "AND status = 'active'"
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, memory_type, content, confidence_score,
                       importance_score, novelty_score, stability_score, confidence_reasons, status,
                       version, source_session, tags, metadata, created_at,
                       updated_at, last_recalled_at, approved_at, last_seen_at, expires_at, embedding, sensitivity, conflicting_memory_id, conflict_reason, last_confirmed_at
                FROM memories
                WHERE user_id = %s {status_clause} AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (user_id, query_embedding, limit),
            )
            return [memory_from_row(row) for row in cursor.fetchall()]

    def record_audit(
        self,
        action: str,
        actor: str,
        memory_id: str,
        previous_value: dict[str, object] | None,
        new_value: dict[str, object] | None,
    ) -> None:
        """Append an audit event outside an existing store mutation transaction."""

        with self.connection.cursor() as cursor:
            self._record_audit(cursor, action, actor, memory_id, previous_value, new_value)
        self.connection.commit()

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


class PineconeVectorIndex:
    """Pinecone adapter for Qwen/DashScope memory embeddings.

    Postgres remains the source of truth for memory records; this adapter stores
    and searches only the vector document needed for cosine-similarity recall.
    """

    def __init__(self, config: Settings = settings) -> None:
        import requests

        self.config = config
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.config.pinecone_api_key and self.config.pinecone_host)

    def upsert_memory(self, memory: Memory) -> None:
        """Index one memory vector in Pinecone."""

        if not self.enabled:
            raise RuntimeError("Pinecone credentials are required")
        if not memory.embedding:
            raise ValueError("memory embedding is required before vector indexing")
        payload = {
            "vectors": [
                {
                    "id": memory.id,
                    "values": memory.embedding,
                    "metadata": {
                        "user_id": memory.user_id,
                        "status": memory.status.value,
                        "content": memory.content,
                        "updated_at": memory.updated_at.isoformat(),
                    },
                }
            ],
            "namespace": self.config.pinecone_namespace,
        }
        self._request("POST", "/vectors/upsert", json=payload)

    def delete_memory(self, memory_id: str) -> None:
        if not self.enabled:
            return
        self._request("POST", "/vectors/delete", json={"ids": [memory_id], "namespace": self.config.pinecone_namespace})

    def list_memory_ids(self) -> list[str]:
        # Pinecone serverless has no cheap full scan API; reconciliation reports missing vectors and deletes known stale ids via Postgres transitions.
        return []

    def update_memory_status(self, memory: Memory) -> None:
        """Update vector metadata after lifecycle changes."""

        if not self.enabled:
            raise RuntimeError("Pinecone credentials are required")
        self._request(
            "POST",
            "/vectors/update",
            json={
                "id": memory.id,
                "namespace": self.config.pinecone_namespace,
                "setMetadata": {"status": memory.status.value, "updated_at": memory.updated_at.isoformat()},
            },
        )

    def search_memory_ids(self, user_id: str, query_embedding: list[float], *, limit: int = 20, include_inactive: bool = False) -> list[str]:
        """Return memory ids ranked by Pinecone vector similarity."""

        if not self.enabled:
            raise RuntimeError("Pinecone credentials are required")
        payload = {
            "vector": query_embedding,
            "topK": limit,
            "includeMetadata": False,
            "namespace": self.config.pinecone_namespace,
            "filter": {"user_id": {"$eq": user_id}} if include_inactive else {"user_id": {"$eq": user_id}, "status": {"$eq": MemoryStatus.ACTIVE.value}},
        }
        data = self._request("POST", "/query", json=payload)
        return [match["id"] for match in data.get("matches", [])]

    def _request(self, method: str, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        response = self.session.request(
            method,
            self.config.pinecone_host.rstrip("/") + path,
            headers={"Api-Key": self.config.pinecone_api_key, "Content-Type": "application/json"},
            json=json,
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else {}


class AliCloudMemoryStore:
    """Production store: ECS-hosted Postgres records plus Pinecone vector recall."""

    def __init__(self, config: Settings = settings) -> None:
        self.records = PostgresMemoryStore(config)
        self.vectors = PineconeVectorIndex(config)

    def migrate(self) -> None:
        self.records.migrate()

    def add_memory_stream_entry(self, entry: MemoryStreamEntry, *, actor: str = "memory-stream") -> MemoryStreamEntry:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memory_stream (id, user_id, content, kind, importance_score, last_accessed_at, created_at, decay_rate, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (entry.id, entry.user_id, entry.content, entry.kind.value, entry.importance_score, entry.last_accessed_at, entry.created_at, entry.decay_rate, entry.status.value, json.dumps(entry.metadata)),
            )
            self._record_audit(cursor, "stream_append", actor, entry.id, None, memory_stream_snapshot(entry))
        self.connection.commit()
        return entry

    def list_memory_stream(self, user_id: str, *, include_inactive: bool = False) -> list[MemoryStreamEntry]:
        status_clause = "" if include_inactive else "AND status = 'active'"
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT id, user_id, content, kind, importance_score, last_accessed_at, created_at, decay_rate, status, metadata
                FROM memory_stream WHERE user_id = %s {status_clause} ORDER BY created_at
                """,
                (user_id,),
            )
            return [memory_stream_from_row(row) for row in cursor.fetchall()]

    def update_memory_stream_access(self, entry_ids: Iterable[str], *, actor: str = "recall") -> None:
        if not entry_ids:
            return
        with self.connection.cursor() as cursor:
            cursor.execute("UPDATE memory_stream SET last_accessed_at = now() WHERE id = ANY(%s)", (list(entry_ids),))
        self.connection.commit()

    def add_memory(self, memory: Memory, *, actor: str = "memory-agent") -> Memory:
        saved = self.records.add_memory(memory, actor=actor)
        self.sync_memory_vector(saved)
        return saved

    def update_memory(self, memory_id: str, *, actor: str = "memory-agent", **changes: object) -> Memory:
        updated = self.records.update_memory(memory_id, actor=actor, **changes)
        self.sync_memory_vector(updated)
        return updated

    def sync_memory_vector(self, memory: Memory) -> None:
        indexable = memory.importance_score >= 0.4 or memory.memory_type in {MemoryType.USER_FACT, MemoryType.PREFERENCE, MemoryType.SEMANTIC, MemoryType.CONVERSATION_SUMMARY} or memory.metadata.get("stream_kind") in {"profile", "fact", "preference", "reflection"}
        if memory.status == MemoryStatus.ACTIVE and indexable:
            self.vectors.upsert_memory(memory)
        else:
            self.vectors.delete_memory(memory.id)

    def vector_search(self, user_id: str, query_embedding: list[float], *, limit: int = 20, include_inactive: bool = False) -> list[Memory]:
        ids = self.vectors.search_memory_ids(user_id, query_embedding, limit=limit, include_inactive=include_inactive)
        memories = []
        for memory_id in ids:
            try:
                memories.append(self.records.get_memory(memory_id))
            except KeyError:
                continue
        return memories

    def __getattr__(self, name: str) -> Any:
        return getattr(self.records, name)


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
    """MinIO/S3-compatible object storage for PDFs, screenshots, and batch files."""

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


def task_record_from_row(row: Iterable[Any]) -> TaskRecord:
    task_id, user_id, title, status, blocker_type, blocker, next_action, evidence, confidence, source, metadata, created_at, updated_at = row
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return TaskRecord(id=task_id, user_id=user_id, title=title, status=TaskRecordStatus(status), blocker_type=blocker_type or "none", blocker=blocker, next_action=next_action, evidence=list(evidence or []), confidence=confidence, source=source or "daily_briefing", metadata=dict(metadata or {}), created_at=created_at, updated_at=updated_at)


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
        approved_at,
        last_seen_at,
        expires_at,
        embedding,
        sensitivity,
        conflicting_memory_id,
        conflict_reason,
        last_confirmed_at,
    ) = row
    if isinstance(tags, str):
        tags = json.loads(tags)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if isinstance(confidence_reasons, str):
        confidence_reasons = json.loads(confidence_reasons)
    if isinstance(embedding, str):
        embedding = json.loads(embedding)
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
        approved_at=approved_at,
        last_seen_at=last_seen_at,
        expires_at=expires_at,
        sensitivity=sensitivity or "low",
        conflicting_memory_id=conflicting_memory_id,
        conflict_reason=conflict_reason,
        last_confirmed_at=last_confirmed_at,
    )


def conflict_from_row(row: Iterable[Any]) -> MemoryConflict:
    return MemoryConflict(
        id=row[0], user_id=row[1], existing_memory_id=row[2], candidate_memory_id=row[3],
        conflict_type=row[4], existing_content=row[5], candidate_content=row[6], status=MemoryConflictStatus(row[7]),
        created_at=row[8], resolved_at=row[9], resolution=row[10],
    )


def memory_stream_from_row(row: Iterable[Any]) -> MemoryStreamEntry:
    entry_id, user_id, content, kind, importance_score, last_accessed_at, created_at, decay_rate, status, metadata = row
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return MemoryStreamEntry(id=entry_id, user_id=user_id, content=content, kind=MemoryStreamKind(kind), importance_score=importance_score, last_accessed_at=last_accessed_at, created_at=created_at, decay_rate=decay_rate, status=MemoryStatus(status), metadata=metadata or {})
