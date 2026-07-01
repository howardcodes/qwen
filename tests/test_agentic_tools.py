from memos_q.agentic.tools import build_tools
from memos_q.store import InMemoryStore

def test_new_open_task_created_and_duplicate_merged():
    store=InMemoryStore(); state={'user_id':'u'}; tools=build_tools(store, state=state)
    first=tools['create_or_update_task'](title='Ship report', next_action='Draft outline')
    second=tools['create_or_update_task'](title='Ship report', status='blocked', blocker='Waiting on data')
    tasks=store.list_task_records('u')
    assert first['ok'] and second['ok']
    assert len(tasks)==1
    assert tasks[0].status.value=='blocked'

def test_send_telegram_requires_permission():
    assert not build_tools(InMemoryStore(), telegram_client=object(), state={'should_notify':False})['send_telegram_message']('hi')['ok']
