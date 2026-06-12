# MemOS-Q: A Self-Evolving Memory Operating System for AI Agents

Designed for QwenCloud, **MemOS-Q** is a production-oriented memory layer that enables AI agents to remember, reason, adapt, and self-correct across conversations, documents, images, and external tools.

Unlike stateless chatbots that rely only on context windows, MemOS-Q introduces persistent memory with confidence scoring, conflict resolution, multimodal understanding, explainable recall, autonomous memory maintenance, and live production integrations.

## What Is Integrated Now?

This repository now includes runnable integration points for the full requested stack. Copy `.env.example` to `.env`, add your API keys, and start the services with Docker Compose.

| Area | Stack | Current implementation |
| --- | --- | --- |
| Frontend | Next.js 12, Tailwind CSS, shadcn/ui-style components, React Flow | `frontend/` dashboard with integration status, memory architecture graph, and live agent chat. |
| Backend | FastAPI, Qwen-Agent | FastAPI app with memory endpoints, QwenCloud chat, Qwen-Agent endpoint, Qwen3-VL ingestion, and integration status. |
| Models | Qwen3.5-Plus, Qwen3.5-Flash, Qwen3-VL-Plus, Qwen Batch API | `QwenCloudClient` calls DashScope/OpenAI-compatible endpoints for reasoning, flash classification, vision extraction, and batch creation. |
| Storage | PostgreSQL, pgvector, Redis, S3-compatible object storage | PostgreSQL/pgvector adapter, Redis cache helper, and S3/MinIO object helper. Docker Compose runs all services. |
| Background jobs | Celery | Celery worker for compaction and Qwen-powered session summarization. |
| Monitoring | Langfuse, OpenTelemetry, Prometheus, Grafana | Langfuse trace decorator, FastAPI OpenTelemetry wiring, `/metrics`, Prometheus scrape config, and Grafana provisioning. |
| Deployment | Docker | API, worker, frontend, Postgres/pgvector, Redis, MinIO, Prometheus, Grafana, and OTel Collector Compose stack. |

## Quick Start: Live Stack

```bash
cp .env.example .env
# Edit .env and replace QWEN_API_KEY, LANGFUSE_* keys, and any storage secrets.
docker compose up --build
```

Services:

- Frontend: <http://localhost:3000>
- FastAPI: <http://localhost:8000/docs>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3001>
- MinIO console: <http://localhost:9001>

## API Key File

All editable credentials live in `.env` (created from `.env.example`). The committed example includes placeholders for:

- `QWEN_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- PostgreSQL password
- Redis URLs
- S3/MinIO access keys
- model names for Qwen3.5-Plus, Qwen3.5-Flash, and Qwen3-VL-Plus

`.env` is ignored by Git so your real keys stay local.

## Core Capabilities

### Explainable Memory

Every recalled memory includes source session, confidence score, timestamp, ranking signals, and reasoning path.

### Self-Correcting Memory

The memory quality engine detects contradictions, outdated information, and superseded preferences. Newer high-confidence memories can deactivate older conflicting memories while preserving an audit trail.

### Multimodal Memory

The `/ingest/vision` endpoint uses Qwen3-VL to extract memory-worthy information from images, screenshots, PDFs, and document URLs.

### Autonomous Memory Maintenance

The Celery worker can run duplicate merging, stable fact promotion, confidence decay, archival, and Qwen-powered session summarization as background jobs.

## Architecture

```text
Next.js Dashboard
   ↓
FastAPI + Qwen-Agent
   ↓
Memory Pipeline
   ├── Retrieval Agent
   ├── Memory Agent
   ├── Profile Agent
   ├── Audit Agent
   └── Celery Compaction Agent
   ↓
QwenCloud Models
   ├── Qwen3.5-Plus
   ├── Qwen3.5-Flash
   ├── Qwen3-VL-Plus
   └── Qwen Batch API
   ↓
PostgreSQL + pgvector / Redis / S3
   ↓
Prometheus + Grafana / OpenTelemetry / Langfuse
```

## Important Endpoints

- `GET /health` — service health.
- `POST /memories` — create an auditable memory.
- `POST /recall` — retrieve memories with explainable scoring.
- `POST /agent/chat` — live QwenCloud-backed memory-aware agent response.
- `POST /agent/qwen-agent` — live Qwen-Agent assistant response.
- `POST /ingest/vision` — Qwen3-VL multimodal memory extraction.
- `POST /users/{user_id}/maintenance` — synchronous maintenance run.
- `GET /integrations/status` — frontend-readable status for configured integrations.
- `GET /metrics` — Prometheus metrics.

## Local Python Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[api,live,monitoring,test]
pytest
uvicorn memos_q.api:app --reload
```

## Example

```python
from memos_q import MemoryOS

memory_os = MemoryOS()

memory_os.remember(
    user_id="user-1",
    content="User prefers concise responses.",
    memory_type="semantic",
    source_session="session-12",
    tags={"preference", "communication"},
)

results = memory_os.recall("user-1", "How should I answer this user?")

for item in results:
    print(item.memory.content)
    print(item.explanation.reasoning_path)
```

## Built for Qwen Code Challenge

MemOS-Q demonstrates QwenCloud-powered memory workflows while addressing a fundamental challenge for next-generation AI systems: **How can AI remember responsibly, transparently, and at scale?**
