# MemOS-Q: A Self-Evolving Memory Operating System for AI Agents

Designed for QwenCloud, **MemOS-Q** is a production-oriented memory layer that enables AI agents to remember, reason, adapt, and self-correct across conversations, documents, images, and external tools.

Unlike stateless chatbots that rely only on context windows, MemOS-Q introduces persistent memory with confidence scoring, conflict resolution, multimodal understanding, explainable recall, autonomous memory maintenance, and live production integrations.

## What Is Integrated Now?

This repository includes runnable integration points for the requested stack. Copy `.env.example` to `.env`, add your keys, and run either the full Docker Compose stack or individual local development processes.

| Area | Stack | Current implementation |
| --- | --- | --- |
| Frontend | Next.js 12, Tailwind CSS, shadcn/ui-style components, React Flow | `frontend/` dashboard with integration status, memory architecture graph, and live agent chat. |
| Backend | FastAPI, Qwen-Agent | FastAPI app with memory endpoints, QwenCloud chat, Qwen-Agent endpoint, Qwen3-VL ingestion, and integration status. |
| Models | Qwen3.5-Plus, Qwen3.5-Flash, Qwen3-VL-Plus, Qwen Batch API | `QwenCloudClient` calls DashScope/OpenAI-compatible endpoints for reasoning, flash classification, vision extraction, and batch creation. |
| Storage | Alibaba Cloud RDS PostgreSQL, OpenSearch Vector Engine, Redis, S3-compatible object storage | RDS/PostgreSQL memory-record adapter, OpenSearch Vector Engine cosine vector recall, Redis cache helper, and S3/OSS object helper. Docker Compose remains available for local service smoke tests. |
| Background jobs | Celery | Celery worker for compaction and Qwen-powered session summarization. |
| Monitoring | Langfuse, OpenTelemetry, Prometheus, Grafana | Langfuse trace decorator, FastAPI OpenTelemetry wiring, `/metrics`, Prometheus scrape config, and Grafana provisioning. |
| Deployment | Docker | API, worker, frontend, Postgres/pgvector, Redis, MinIO, Prometheus, Grafana, and OTel Collector Compose stack. |

## Prerequisites

Install the tools that match the way you want to run the project:

- **Docker path:** Docker Engine with Docker Compose v2.
- **Local backend path:** Python 3.10+.
- **Local frontend path:** Node.js 20+ and npm.
- **Live model calls:** a QwenCloud/DashScope API key.
- **Langfuse tracing:** Langfuse public and secret keys, if you want traces in Langfuse.

## Configure Secrets and Runtime Settings

All editable credentials live in `.env`, which is intentionally ignored by Git.

```bash
cp .env.example .env
```

Then edit `.env` and replace placeholders such as:

```bash
QWEN_API_KEY=replace-with-your-qwen-api-key
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_EMBEDDING_DIMENSIONS=1024
# Set true in production to fail closed if Alibaba/Qwen embedding calls fail.
QWEN_REQUIRE_LIVE_EMBEDDINGS=true
LANGFUSE_PUBLIC_KEY=replace-with-langfuse-public-key
LANGFUSE_SECRET_KEY=replace-with-langfuse-secret-key
POSTGRES_DSN=postgresql://memos:replace-with-rds-password@rm-xxxx.pg.rds.aliyuncs.com:5432/memos
OPENSEARCH_ENDPOINT=https://opensearch-xxxx.aliyuncs.com
OPENSEARCH_USERNAME=replace-with-opensearch-username
OPENSEARCH_PASSWORD=replace-with-opensearch-password
S3_ACCESS_KEY_ID=replace-with-oss-access-key-id
S3_SECRET_ACCESS_KEY=replace-with-oss-secret-access-key
```

Use these storage modes:

```bash
# Production on Alibaba Cloud: ECS runs the API/worker, RDS stores memory
# records/audit history, Qwen Embedding creates vectors, and OpenSearch Vector
# Engine searches those vectors with cosine similarity.
MEMOS_STORE=alicloud
QWEN_REQUIRE_LIVE_EMBEDDINGS=true

# Development/test-only in-memory store. Do not use for production.
MEMOS_STORE=memory

# Development-only PostgreSQL store without OpenSearch Vector Engine.
MEMOS_STORE=postgres
```

## Run Everything with Docker Compose

This is the intended full-stack execution path.

```bash
cp .env.example .env
# Edit .env with your real QwenCloud/Langfuse/storage values.
docker compose up --build
```

Open the services:

- Frontend dashboard: <http://localhost:3000>
- FastAPI docs: <http://localhost:8000/docs>
- FastAPI health: <http://localhost:8000/health>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3001>
- MinIO console: <http://localhost:9001>

Useful Docker commands:

```bash
# Check whether the API container is running and healthy.
docker compose ps api

# Follow API startup logs if localhost:8000 refuses connections.
docker compose logs -f api

# Rebuild only the API after backend edits.
docker compose up --build api

# Run the Celery worker with the rest of the stack.
docker compose up --build worker

# View app logs.
docker compose logs -f api worker frontend

# Stop services but keep volumes.
docker compose down

# Stop services and remove Postgres/MinIO/Grafana volumes.
docker compose down -v
```

The API service waits for healthy PostgreSQL and Redis containers before it starts, and the frontend waits for the API health check before starting. If the browser reports `POST http://localhost:8000/agent/chat net::ERR_CONNECTION_REFUSED`, the API container is not reachable; run `docker compose ps api` and `docker compose logs -f api` first.

## Run the Backend Locally

Use this mode when you want FastAPI running on your machine instead of in Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api,live,monitoring,test]'
cp .env.example .env
```

For local development without PostgreSQL, set `MEMOS_STORE=memory` in `.env`, then run:

```bash
uvicorn memos_q.api:app --host 0.0.0.0 --port 8000 --reload
```

For local FastAPI with Docker-managed Postgres/Redis/MinIO, run the backing services first and then start Uvicorn:

```bash
docker compose up -d postgres redis minio otel-collector prometheus grafana
MEMOS_STORE=postgres \
POSTGRES_DSN=postgresql://memos:memos@localhost:5432/memos \
REDIS_URL=redis://localhost:6379/0 \
S3_ENDPOINT_URL=http://localhost:9000 \
uvicorn memos_q.api:app --host 0.0.0.0 --port 8000 --reload
```

## Run the Frontend Locally

In a second terminal, install and run the Next.js dashboard:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Then open <http://localhost:3000>.

## Run Celery Locally

Start Redis first, then run the worker from the repository root:

```bash
docker compose up -d redis
source .venv/bin/activate
CELERY_BROKER_URL=redis://localhost:6379/1 \
CELERY_RESULT_BACKEND=redis://localhost:6379/2 \
celery -A memos_q.workers.celery_app.celery_app worker --loglevel=INFO
```

If you want scheduled jobs from the configured Celery beat schedule, run beat in another terminal:

```bash
source .venv/bin/activate
CELERY_BROKER_URL=redis://localhost:6379/1 \
CELERY_RESULT_BACKEND=redis://localhost:6379/2 \
celery -A memos_q.workers.celery_app.celery_app beat --loglevel=INFO
```

## API Smoke Tests

After the API is running, use these commands to verify the core flow.

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/memories \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: demo-user' \
  -d '{
    "content": "User prefers concise answers about AI agents.",
    "memory_type": "preference",
    "source_session": "manual-smoke-test",
    "tags": ["preference", "agents"]
  }'
```

```bash
curl -X POST http://localhost:8000/recall \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: demo-user' \
  -d '{
    "query": "How should I answer this user about agents?",
    "limit": 3
  }'
```

```bash
curl http://localhost:8000/integrations/status
```

Live QwenCloud calls require a real `QWEN_API_KEY` in `.env`:

```bash
curl -X POST http://localhost:8000/agent/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: demo-user' \
  -d '{
    "message": "What do you remember about me?",
    "source_session": "live-qwen-smoke-test"
  }'
```

Qwen3-VL ingestion requires a reachable image or document URL:

```bash
curl -X POST http://localhost:8000/ingest/vision \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "demo-user",
    "image_url": "https://example.com/diagram.png",
    "source_session": "vision-smoke-test",
    "prompt": "Extract durable project memory facts from this image."
  }'
```

## Run Tests and Checks

```bash
pytest -q
```

```bash
python -m compileall src tests
```

```bash
node -c frontend/next.config.js
node -c frontend/postcss.config.js
node -c frontend/tailwind.config.js
```

```bash
python - <<'PY'
import tomllib
from pathlib import Path
with Path('pyproject.toml').open('rb') as file:
    tomllib.load(file)
print('pyproject ok')
PY
```

When npm registry access is available, you can also validate the frontend dependency graph and production build:

```bash
cd frontend
npm install
npm run build
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

## Python API Example

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

## Troubleshooting

- If `docker compose` fails because `.env` is missing, run `cp .env.example .env` first.
- If `localhost:3000` shows `POST http://localhost:8000/agent/chat net::ERR_CONNECTION_REFUSED`, the FastAPI container is down or unhealthy. Run `docker compose ps api` and `docker compose logs -f api`, then rebuild with `docker compose up --build api`.
- If the API starts locally but tries to connect to Postgres, set `MEMOS_STORE=memory` in `.env`.
- If `/agent/chat` fails with `QWEN_API_KEY is required`, add a real QwenCloud key to `.env`.
- If frontend requests fail, confirm `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and that FastAPI is running.
- If Grafana logs `stat /var/lib/grafana/dashboards: no such file or directory`, rebuild with the latest compose file; it mounts `./monitoring/grafana/dashboards` into that path.
- If npm install fails with a registry `403`, fix npm registry/auth settings and rerun `npm install` inside `frontend/`.

## Built for Qwen Code Challenge

MemOS-Q demonstrates QwenCloud-powered memory workflows while addressing a fundamental challenge for next-generation AI systems: **How can AI remember responsibly, transparently, and at scale?**
