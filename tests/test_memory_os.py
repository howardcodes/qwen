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

    assert result.action == "created"
    assert result.memory.status == MemoryStatus.ACTIVE
    assert result.memory.id in vectors.ids


def test_duplicate_structured_memory_does_not_create_duplicate_row():
    from memos_q.engine import MemoryCandidate
    memory_os = build_memory_os()
    memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s1")
    result = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("My name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s2")

    assert result.action == "duplicate"
    assert len(memory_os.inspect("user-1", include_inactive=True)) == 1


def test_conflict_confirmation_accept_and_reject_paths_sync_vectors():
    from memos_q.engine import MemoryCandidate
    vectors = RecordingVectorIndex()
    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=vectors, fallback_embedding_dimensions=2)
    old = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Joshua.", "user_fact", "profile.name", "Joshua", 0.98), source_session="s1").memory
    conflict = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s2")

    assert conflict.action == "conflict"
    assert conflict.memory.status == MemoryStatus.PENDING_CONFLICT_CONFIRMATION
    assert "Joshua" in conflict.prompt and "Mark" in conflict.prompt
    assert conflict.memory.id not in vectors.ids

    accepted = memory_os.resolve_pending_conflict("user-1", "Yes update it")
    memories = {m.id: m for m in memory_os.inspect("user-1", include_inactive=True)}
    assert accepted.action == "accepted_candidate"
    assert memories[old.id].status == MemoryStatus.SUPERSEDED
    assert memories[conflict.memory.id].status == MemoryStatus.ACTIVE
    assert old.id not in vectors.ids and conflict.memory.id in vectors.ids

    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=RecordingVectorIndex(), fallback_embedding_dimensions=2)
    old = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Joshua.", "user_fact", "profile.name", "Joshua", 0.98), source_session="s1").memory
    conflict = memory_os.ingest_candidate(user_id="user-1", candidate=MemoryCandidate("User's name is Mark.", "user_fact", "profile.name", "Mark", 0.98), source_session="s2")
    rejected = memory_os.resolve_pending_conflict("user-1", "No, keep Joshua")
    memories = {m.id: m for m in memory_os.inspect("user-1", include_inactive=True)}
    assert rejected.action == "kept_existing"
    assert memories[old.id].status == MemoryStatus.ACTIVE
    assert memories[conflict.memory.id].status == MemoryStatus.REJECTED


def test_reconciliation_fixes_missing_and_stale_vectors():
    vectors = RecordingVectorIndex()
    memory_os = MemoryOS(embedding_provider=RecordingEmbeddingProvider(), vector_index=vectors, fallback_embedding_dimensions=2)
    active = memory_os.remember(user_id="user-1", content="User prefers Python.", source_session="s1")
    vectors.ids = {"stale-id"}

    report = memory_os.reconcile_vectors()

    assert active.id in vectors.ids
    assert "stale-id" not in vectors.ids
    assert report["fixed"] == 2
