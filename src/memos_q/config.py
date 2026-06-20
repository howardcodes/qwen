"""Runtime configuration for live MemOS-Q deployments.

Values are loaded from the environment and, when present, from an editable
``.env`` file at the repository root. The parser is intentionally small so the
core package can still be imported in tests before optional production
services are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from a dotenv-style file without overwriting env."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_DEFAULTS = {
    "environment": "development",
    "api_base_url": "http://localhost:8000",
    "frontend_url": "http://localhost:3000",
    "memos_store": "memory",
    "pinecone_api_key": "",
    "pinecone_host": "",
    "pinecone_namespace": "memos-q",
    "pinecone_index": "memos-q-vectors",
    "qwen_api_key": "",
    "qwen_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen_reasoning_model": "qwen3.5-plus",
    "qwen_flash_model": "qwen3.5-flash",
    "qwen_vl_model": "qwen3-vl-plus",
    "qwen_embedding_model": "text-embedding-v4",
    "qwen_embedding_dimensions": "1024",
    "postgres_dsn": "postgresql://memos:memos@localhost:5432/memos",
    "redis_url": "redis://localhost:6379/0",
    "s3_endpoint_url": "http://localhost:9000",
    "s3_access_key_id": "memos",
    "s3_secret_access_key": "memos-password",
    "s3_bucket": "memos-q",
    "s3_region": "us-east-1",
    "celery_broker_url": "redis://localhost:6379/1",
    "celery_result_backend": "redis://localhost:6379/2",
    "langfuse_public_key": "",
    "langfuse_secret_key": "",
    "langfuse_host": "https://cloud.langfuse.com",
    "otel_exporter_otlp_endpoint": "http://otel-collector:4317",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """All configurable endpoints, credentials, and model names."""

    environment: str = "development"
    api_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    memos_store: str = "memory"

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_reasoning_model: str = "qwen3.5-plus"
    qwen_flash_model: str = "qwen3.5-flash"
    qwen_vl_model: str = "qwen3-vl-plus"
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_embedding_dimensions: int = 1024
    qwen_require_live_embeddings: bool = False

    pinecone_api_key: str = ""
    pinecone_host: str = ""
    pinecone_namespace: str = "memos-q"
    pinecone_index: str = "memos-q-vectors"

    postgres_dsn: str = "postgresql://memos:memos@localhost:5432/memos"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key_id: str = "memos"
    s3_secret_access_key: str = "memos-password"
    s3_bucket: str = "memos-q"
    s3_region: str = "us-east-1"

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        """Create settings from ``env_file`` and process environment values."""

        load_env_file(env_file)
        return cls(
            environment=os.getenv("MEMOS_ENV", _DEFAULTS["environment"]),
            api_base_url=os.getenv("API_BASE_URL", _DEFAULTS["api_base_url"]),
            frontend_url=os.getenv("FRONTEND_URL", _DEFAULTS["frontend_url"]),
            memos_store=os.getenv("MEMOS_STORE", _DEFAULTS["memos_store"]),
            qwen_api_key=os.getenv("QWEN_API_KEY", _DEFAULTS["qwen_api_key"]),
            qwen_base_url=os.getenv("QWEN_BASE_URL", _DEFAULTS["qwen_base_url"]),
            qwen_reasoning_model=os.getenv("QWEN_REASONING_MODEL", _DEFAULTS["qwen_reasoning_model"]),
            qwen_flash_model=os.getenv("QWEN_FLASH_MODEL", _DEFAULTS["qwen_flash_model"]),
            qwen_vl_model=os.getenv("QWEN_VL_MODEL", _DEFAULTS["qwen_vl_model"]),
            qwen_embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", _DEFAULTS["qwen_embedding_model"]),
            qwen_embedding_dimensions=int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", _DEFAULTS["qwen_embedding_dimensions"])),
            qwen_require_live_embeddings=os.getenv("QWEN_REQUIRE_LIVE_EMBEDDINGS", "false").lower() == "true",
            pinecone_api_key=os.getenv("PINECONE_API_KEY", _DEFAULTS["pinecone_api_key"]),
            pinecone_host=os.getenv("PINECONE_HOST", _DEFAULTS["pinecone_host"]),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", _DEFAULTS["pinecone_namespace"]),
            pinecone_index=os.getenv("PINECONE_INDEX", _DEFAULTS["pinecone_index"]),
            postgres_dsn=os.getenv("POSTGRES_DSN", _DEFAULTS["postgres_dsn"]),
            redis_url=os.getenv("REDIS_URL", _DEFAULTS["redis_url"]),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", _DEFAULTS["s3_endpoint_url"]),
            s3_access_key_id=os.getenv("S3_ACCESS_KEY_ID", _DEFAULTS["s3_access_key_id"]),
            s3_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", _DEFAULTS["s3_secret_access_key"]),
            s3_bucket=os.getenv("S3_BUCKET", _DEFAULTS["s3_bucket"]),
            s3_region=os.getenv("S3_REGION", _DEFAULTS["s3_region"]),
            celery_broker_url=os.getenv("CELERY_BROKER_URL", _DEFAULTS["celery_broker_url"]),
            celery_result_backend=os.getenv("CELERY_RESULT_BACKEND", _DEFAULTS["celery_result_backend"]),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", _DEFAULTS["langfuse_public_key"]),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", _DEFAULTS["langfuse_secret_key"]),
            langfuse_host=os.getenv("LANGFUSE_HOST", _DEFAULTS["langfuse_host"]),
            otel_exporter_otlp_endpoint=os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                _DEFAULTS["otel_exporter_otlp_endpoint"],
            ),
        )


settings = Settings.from_env()
