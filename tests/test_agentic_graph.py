from memos_q.agentic.graph import run_agentic
from memos_q.models import ChatTurn, TaskRecord, TaskRecordStatus
from memos_q.store import InMemoryStore

class FakeQwen:
    def __init__(self, text='{"goal":"g","plan":[{"step_id":"s1","tool":"list_open_tasks","reason":"r","args":{}}],"success_criteria":["ok"],"risk_level":"low"}'):
        self.text=text
    def chat(self, *args, **kwargs): return self.text

def test_graph_loads_context_and_completes_successfully():
    store=InMemoryStore(); store.append_conversation_turn('u','c',ChatTurn('user','Need to ship report'))
    result=run_agentic(store, FakeQwen(), user_id='u', input_text='what next')
    assert result['final_briefing']
    assert result['observations']
    assert store.list_agent_runs('u')

def test_unknown_tool_is_rejected_and_graph_does_not_crash():
    result=run_agentic(InMemoryStore(), FakeQwen('{"plan":[{"tool":"shell","args":{}}]}'), user_id='u')
    assert result['final_briefing']
    assert all(o.get('tool') != 'shell' for o in result['observations'])

def test_invalid_qwen_json_is_handled_safely():
    result=run_agentic(InMemoryStore(), FakeQwen('not json'), user_id='u')
    assert result['final_briefing']
    assert result['errors']

def test_replan_limit_is_enforced():
    result=run_agentic(InMemoryStore(), FakeQwen('{"plan":[{"step_id":"s1","tool":"send_telegram_message","args":{"text":"x"}}]}'), user_id='u')
    assert result['metadata']['replans'] <= 1
    assert result['metadata']['planning_rounds'] <= 2
