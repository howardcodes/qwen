from memos_q.daily_summary import generate_daily_summary_text, run_daily_summary
from memos_q.models import ChatTurn
from memos_q.store import InMemoryStore
from memos_q.telegram import TelegramClient, truncate_telegram_message


class FakeQwen:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '[{"title":"Review Telegram settings","status":"open","blocker_type":"none","blocker":null,"next_action":"Review Telegram settings","evidence":["Telegram daily summaries sound useful."],"confidence":0.8}]'
        return "Topics\n- Qwen planning\n- Qwen planning\nNext\n- Review Telegram settings"


class RecordingTelegram(TelegramClient):
    def __init__(self):
        super().__init__(bot_token="token", chat_id="chat")
        self.messages = []

    def send_message(self, text, *, chat_id=None):
        self.messages.append((text, chat_id))
        return type("Result", (), {"sent": True, "error_message": None})()


def test_daily_summary_generation_deduplicates_topics():
    store = InMemoryStore()
    store.append_conversation_turn("u1", "c1", ChatTurn("user", "Let's discuss Qwen memory summaries."))
    store.append_conversation_turn("u1", "c1", ChatTurn("assistant", "Telegram daily summaries sound useful."))

    summary = generate_daily_summary_text("u1", store, FakeQwen())

    assert summary.count("Qwen planning") == 1
    assert "Review Telegram settings" in summary


def test_run_daily_summary_persists_and_sends():
    store = InMemoryStore()
    store.append_conversation_turn("u1", "c1", ChatTurn("user", "Remember I prefer concise morning updates."))
    telegram = RecordingTelegram()

    summary = run_daily_summary("u1", store, FakeQwen(), telegram)

    assert summary.sent_to_telegram is True
    assert summary.sent_at is not None
    assert telegram.messages
    assert store.list_daily_summaries("u1") == [summary]


def test_telegram_missing_env_vars_fails_gracefully(monkeypatch):
    client = TelegramClient(bot_token="", chat_id="")

    result = client.send_message("hello")

    assert result.sent is False
    assert "not configured" in result.error_message


def test_empty_summary_is_not_sent_or_marked_sent():
    class EmptyQwen:
        def chat(self, messages, **kwargs):
            return ""

    store = InMemoryStore()
    store.append_conversation_turn("u1", "c1", ChatTurn("user", "Hello"))
    telegram = RecordingTelegram()

    summary = run_daily_summary("u1", store, EmptyQwen(), telegram)

    assert summary.sent_to_telegram is False
    assert telegram.messages == []
    assert "empty" in summary.error_message.lower()


def test_truncate_telegram_message_keeps_limit():
    assert len(truncate_telegram_message("x" * 5000)) <= 4096


def test_daily_briefing_extracts_and_persists_task_records():
    store = InMemoryStore()
    store.append_conversation_turn("u1", "c1", ChatTurn("user", "I need to finish the release plan but I am waiting on API credentials."))

    summary = generate_daily_summary_text("u1", store, FakeQwen())

    tasks = store.list_task_records("u1")
    assert len(tasks) == 1
    assert tasks[0].title == "Review Telegram settings"
    assert tasks[0].next_action == "Review Telegram settings"
    assert "Review Telegram settings" in summary
