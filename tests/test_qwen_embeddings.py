from memos_q.config import Settings
from memos_q.integrations.qwen_cloud import QwenCloudClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
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
