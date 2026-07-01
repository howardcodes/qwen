from memos_q.agentic.memory import persist_memory_updates
from memos_q.store import InMemoryStore

def test_memory_proposal_does_not_persist_unsupported_short_facts():
    store=InMemoryStore()
    saved=persist_memory_updates({'user_id':'u','observations':[{'data':{'memory_candidate':'temporary'}}]}, store)
    assert saved == []
    assert store.list_memories('u') == []
