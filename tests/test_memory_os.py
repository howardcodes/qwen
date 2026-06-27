from memos_q import MemoryOS
from memos_q.models import MemoryStatus, MemoryType, RelationType
from memos_q.store import InMemoryStore


class RecordingEmbeddingProvider:
    def __init__(self):
        self.texts = []

    def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        if "Python" in text or "concise" in text or "MacBook" in text or "ThinkPad" in text or "hackathon" in text:
            return [1.0, 0.0]
        if "Rust" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def build_memory_os() -> MemoryOS:
    return MemoryOS(embedding_provider=RecordingEmbeddingProvider(), fallback_embedding_dimensions=2)


def test_recall_returns_explainable_ranking():
    memory_os = build_memory_os()
    memory_os.remember(
        user_id="user-1",
        content="User prefers concise responses when discussing AI agents.",
        memory_type="semantic",
        source_session="session-12",
        tags={"preference", "agents"},
        confidence_score=0.94,
    )

    results = memory_os.recall("user-1", "How should I answer questions about AI agents?")

    assert results
    assert results[0].memory.content.startswith("User prefers concise")
    assert results[0].explanation.source_session == "session-12"
    assert results[0].explanation.confidence_score == 0.94
    assert "semantic" in results[0].explanation.ranking_signals
    assert results[0].explanation.reasoning_path


def test_newer_confident_memory_supersedes_conflicting_memory():
    memory_os = build_memory_os()
    old = memory_os.remember(
        user_id="user-1",
        content="User uses MacBook M3.",
        memory_type="semantic",
        source_session="session-1",
        confidence_score=0.7,
    )
    new = memory_os.remember(
        user_id="user-1",
        content="User uses ThinkPad X1.",
        memory_type="semantic",
        source_session="session-2",
        confidence_score=0.95,
    )

    memories = {memory.id: memory for memory in memory_os.inspect("user-1", include_inactive=True)}
    assert memories[old.id].status == MemoryStatus.DEPRECATED
    assert memories[new.id].status == MemoryStatus.ACTIVE
    edges = memory_os.store.edges_for(new.id)
    assert {edge.relation_type for edge in edges} >= {RelationType.CONTRADICTS, RelationType.SUPERSEDES}


def test_maintenance_merges_duplicates_and_promotes_stable_facts():
    memory_os = build_memory_os()
    first = memory_os.remember(
        user_id="user-1",
        content="User prefers Python for AI agents.",
        memory_type="episodic",
        source_session="session-1",
        stability_score=0.85,
        confidence_score=0.9,
    )
    duplicate = memory_os.remember(
        user_id="user-1",
        content="User prefers Python for AI agents.",
        memory_type="episodic",
        source_session="session-2",
        stability_score=0.85,
        confidence_score=0.8,
    )

    report = memory_os.maintenance("user-1")
    memories = {memory.id: memory for memory in memory_os.inspect("user-1", include_inactive=True)}

    assert report["merged"] == 1
    assert memories[duplicate.id].status == MemoryStatus.ARCHIVED
    assert memories[first.id].memory_type == MemoryType.SEMANTIC


def test_forget_keeps_audit_history():
    memory_os = build_memory_os()
    memory = memory_os.remember(
        user_id="user-1",
        content="User is preparing for a hackathon.",
        source_session="session-3",
    )

    memory_os.forget("user-1", memory.id)

    assert memory_os.inspect("user-1") == []
    assert any(event.action == "update" for event in memory_os.store.audit_log(memory.id))

def test_memory_os_uses_embedding_provider_for_remember_and_recall():
    provider = RecordingEmbeddingProvider()
    memory_os = MemoryOS(embedding_provider=provider, fallback_embedding_dimensions=2)
    memory_os.remember(user_id="user-1", content="User prefers Python.", source_session="session-1")
    memory_os.remember(user_id="user-1", content="Project uses Rust.", source_session="session-2")

    results = memory_os.recall("user-1", "Python", limit=1)

    assert results[0].memory.content == "User prefers Python."
    assert "vector" in results[0].explanation.ranking_signals
    assert "Python" in provider.texts


def test_pending_review_recall_uses_record_fallback_when_vector_index_misses():
    class MissingPendingVectorStore(InMemoryStore):
        def vector_search(self, user_id, query_embedding, *, limit=20, include_inactive=False):
            return [memory for memory in super().vector_search(user_id, query_embedding, limit=limit, include_inactive=include_inactive) if memory.status == MemoryStatus.ACTIVE]

    memory_os = MemoryOS(store=MissingPendingVectorStore(), embedding_provider=RecordingEmbeddingProvider(), fallback_embedding_dimensions=2)
    memory_os.remember(user_id="user-1", content="User's name is Mark", source_session="session-name", status=MemoryStatus.PENDING_REVIEW)
    memory_os.remember(user_id="user-1", content="User likes to play badminton", source_session="session-sport")

    contents = [result.memory.content for result in memory_os.recall("user-1", "What is my name?", include_pending_review=True)]

    assert "User's name is Mark" in contents

class RecordingVectorIndex:
    def __init__(self):
        self.ids = set()
        self.upserts = []
        self.deletes = []
    def upsert_memory(self, memory):
        self.ids.add(memory.id); self.upserts.append(memory.id)
    def delete_memory(self, memory_id):
        self.ids.discard(memory_id); self.deletes.append(memory_id)
    def list_memory_ids(self):
        return sorted(self.ids)


def test_structured_non_conflicting_memory_auto_active_and_upserted():
    vectors = RecordingVectorIndex()
    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=vectors, fallback_embedding_dimensions=2)

    result = memory_os.ingest_candidate(
        user_id="user-1",
        candidate=__import__("memos_q.engine", fromlist=["MemoryCandidate"]).MemoryCandidate(
            content="User's name is Mark.", type="user_fact", key="profile.name", value="Mark", confidence=0.98, sensitivity="low"
        ),
        source_session="session-1",
    )

    assert result.action == "profile_updated"
    assert result.memory is None
    assert memory_os.get_user_profile("user-1").name == "Mark"
    assert not vectors.ids


def test_duplicate_structured_memory_does_not_create_duplicate_row():
    from memos_q.engine import MemoryCandidate
    memory_os = build_memory_os()
    memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s1")
    result = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("My name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s2")

    assert result.action == "profile_updated"
    assert memory_os.get_user_profile("user-1").name == "Mark"
    assert len(memory_os.inspect("user-1", include_inactive=True)) == 0


def test_profile_facts_update_structured_profile_without_vector_conflict_rows():
    from memos_q.engine import MemoryCandidate
    vectors = RecordingVectorIndex()
    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=vectors, fallback_embedding_dimensions=2)

    first = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Joshua.", "user_fact", "profile.name", "Joshua", 0.98), source_session="s1")
    second = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s2")

    assert first.action == "profile_updated"
    assert second.action == "profile_updated"
    assert memory_os.get_user_profile("user-1").name == "Mark"
    assert memory_os.inspect("user-1", include_inactive=True) == []
    assert not vectors.ids


def test_reconciliation_fixes_missing_and_stale_vectors():
    vectors = RecordingVectorIndex()
    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=vectors, fallback_embedding_dimensions=2)
    active = memory_os.remember(user_id="user-1", content="User prefers Python.", source_session="s1")
    vectors.ids = {"stale-id"}

    report = memory_os.reconcile_vectors()

    assert active.id in vectors.ids
    assert "stale-id" not in vectors.ids
    assert report["fixed"] == 2


def test_recall_searches_memory_stream_observations_without_hardcoded_extractors():
    from memos_q.models import MemoryStreamEntry, MemoryStreamKind

    memory_os = build_memory_os()
    memory_os.store.add_memory_stream_entry(
        MemoryStreamEntry(
            user_id="user-1",
            content="User said: Hello, my name is Morgan and I enjoy squash.",
            kind=MemoryStreamKind.OBSERVATION,
            importance_score=5,
            metadata={"source_session": "chat-1"},
        )
    )

    results = memory_os.recall("user-1", "What is my name?", limit=3)

    assert results
    assert results[0].memory.metadata["stream_entry_id"]
    assert "Morgan" in results[0].memory.content


def test_postgres_store_exposes_public_record_audit_for_workers():
    from memos_q.integrations.durable import PostgresMemoryStore

    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            calls.append((sql, params))

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            calls.append(("commit", None))

    store = PostgresMemoryStore.__new__(PostgresMemoryStore)
    store.connection = FakeConnection()

    store.record_audit("job_start", "celery-memory-evolution", "user-1", None, {"task_id": "task-1"})

    assert any("INSERT INTO audit_log" in sql for sql, _ in calls if isinstance(sql, str))
    assert ("commit", None) in calls


def test_json_file_store_persists_memories_across_instances(tmp_path):
    from memos_q.store import JsonFileMemoryStore

    store_path = tmp_path / "memory-store.json"
    memory_os = MemoryOS(store=JsonFileMemoryStore(store_path), embedding_provider=RecordingEmbeddingProvider(), fallback_embedding_dimensions=2)
    memory_os.remember(user_id="user-1", content="User prefers Baskerville typography.", source_session="session-1")
    memory_os.store.upsert_user_profile("user-1", name="Morgan")

    reloaded = MemoryOS(store=JsonFileMemoryStore(store_path), embedding_provider=RecordingEmbeddingProvider(), fallback_embedding_dimensions=2)

    assert reloaded.recall("user-1", "What typography does the user prefer?", limit=1)[0].memory.content == "User prefers Baskerville typography."
    assert reloaded.get_user_profile("user-1").name == "Morgan"


def test_recall_rejects_unrelated_memory_despite_high_importance_and_recency():
    memory_os = build_memory_os()
    memory_os.remember(
        user_id="user-1",
        content="User strongly prefers zygomorphic orchid taxonomy notes.",
        source_session="profile",
        importance_score=1.0,
    )

    results = memory_os.recall("user-1", "voltage regulator solder bridge", limit=5)

    assert results == []


def test_raw_chat_observations_are_not_recalled_as_long_term_memory():
    from memos_q.models import MemorySource, MemoryStreamEntry, MemoryStreamKind

    memory_os = build_memory_os()
    memory_os.store.add_memory_stream_entry(
        MemoryStreamEntry(
            user_id="user-1",
            content="User said: EntityA EntityB EntityC were discussed.",
            kind=MemoryStreamKind.OBSERVATION,
            importance_score=10,
            metadata={"source_session": "chat", "source": MemorySource.RAW_CHAT_LOG.value},
        )
    )

    assert memory_os.recall("user-1", "EntityA", limit=3) == []
