from memos_q.config import Settings
from memos_q.integrations.qwen_cloud import QwenCloudClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }

    def iter_lines(self, decode_unicode=False):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        return iter(lines)


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout, stream=False):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout, "stream": stream})
        return FakeResponse()


def test_qwen_embeddings_use_openai_compatible_dashscope_endpoint():
    config = Settings(qwen_api_key="test-key", qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    client = QwenCloudClient(config)
    fake_session = FakeSession()
    client.session = fake_session

    embeddings = client.embed_texts(["alpha", "beta"], dimensions=2)

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    call = fake_session.calls[0]
    assert call["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert call["json"] == {
        "model": "text-embedding-v4",
        "input": ["alpha", "beta"],
        "dimensions": 2,
        "encoding_format": "float",
    }
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_qwen_chat_stream_yields_openai_compatible_delta_tokens():
    config = Settings(qwen_api_key="test-key", qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    client = QwenCloudClient(config)
    fake_session = FakeSession()
    client.session = fake_session

    tokens = list(client.chat_stream([{"role": "user", "content": "hello"}], model="qwen-test"))

    assert tokens == ["Hel", "lo"]
    call = fake_session.calls[0]
    assert call["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert call["stream"] is True
    assert call["json"]["stream"] is True
    assert call["json"]["model"] == "qwen-test"

from memos_q.config import Settings
from memos_q.integrations.qwen_cloud import QwenCloudClient, QwenMessage


def test_qwen_chat_payload_uses_flash_default_and_max_tokens():
    client = QwenCloudClient(Settings(qwen_chat_default_model="qwen3.5-flash", qwen_chat_max_tokens=800))
    payload = client._chat_payload([QwenMessage("user", "hi")], model=None, temperature=0.1, max_tokens=None)

    assert payload["model"] == "qwen3.5-flash"
    assert payload["max_tokens"] == 800


def test_qwen_chat_payload_allows_task_specific_token_limit():
    client = QwenCloudClient(Settings(qwen_chat_default_model="qwen3.5-flash", qwen_memory_extraction_max_tokens=500))
    payload = client._chat_payload([QwenMessage("user", "extract")], model="qwen3.5-flash", temperature=0, max_tokens=300)

    assert payload["model"] == "qwen3.5-flash"
    assert payload["max_tokens"] == 300
