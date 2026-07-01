from memos_q.agentic.notification import decide_notification

def test_blocked_task_causes_notification():
    assert decide_notification({'open_tasks':[{'status':'blocked'}]}).should_notify

def test_no_meaningful_update_skips_telegram():
    d=decide_notification({'recent_conversations':[], 'open_tasks':[]})
    assert not d.should_notify and 'Skipped' in d.reason

def test_force_send_sends_anyway():
    assert decide_notification({'metadata':{'force_send':True}}).should_notify
