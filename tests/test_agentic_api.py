import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from memos_q.api import app

def test_agentic_run_returns_plan_observations_and_final_briefing(monkeypatch):
    from memos_q import api
    monkeypatch.setattr(api.qwen_client, 'chat', lambda *a, **k: '{"plan":[{"step_id":"s1","tool":"list_open_tasks","reason":"r","args":{}}]}')
    res=TestClient(app).post('/agentic/run', headers={'x-user-id':'u'}, json={'user_id':'u','trigger':'manual_api_run','input_text':'Check next','send':False})
    assert res.status_code == 200
    data=res.json()
    assert 'plan' in data and 'observations' in data and data['final_briefing']
