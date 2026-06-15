from memos_q import MemoryOS
from memos_q.models import MemoryStatus, MemoryType, RelationType


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
