# MemOS-Q

MemOS-Q is a Qwen-powered memory operating system for AI agents. It pairs a focused Next.js chat UI with a FastAPI backend that can persist memories, recall relevant context, stream Qwen responses, ingest multimodal inputs, run background maintenance, and expose production observability.

## Current Project Snapshot

| Layer | Current implementation |
| --- | --- |
| Frontend | Next.js 12 + React 18 + Tailwind chat page in `frontend/`, centered `MemOS-Q` header, streaming chat composer, persisted demo user ID, and lightweight response formatting for Markdown-style emphasis, code, lists, and quotes. |
| Backend API | FastAPI app in `src/memos_q/api.py` with health, memory CRUD/review, recall, streaming chat, Qwen-Agent, Qwen3-VL ingestion, maintenance, vector reconciliation, integration status, and Prometheus metrics endpoints. |
| Memory engine | `MemoryOS` supports durable records, conversation turns, profile/session state, conflict handling, explainable recall, maintenance, scoring, and fallback embeddings for tests/development. |
| Model integrations | DashScope/OpenAI-compatible Qwen client for chat streaming, reasoning/classification, embeddings, Qwen3-VL extraction, and optional Qwen-Agent integration. |
| Storage modes | JSON file store by default for local development, in-memory test store, PostgreSQL store, and Alibaba-oriented store with PostgreSQL records plus Pinecone vector recall, Redis, and MinIO/S3 helpers. |
| Background jobs | Celery worker app for memory compaction and Qwen-powered conversation summarization. |
| Observability | Prometheus `/metrics`, optional OpenTelemetry FastAPI instrumentation, Langfuse tracing hooks, Prometheus config, Grafana provisioning, and an OpenTelemetry Collector config. |
| Deployment | Dockerfiles for API, worker, and frontend plus `docker-compose.yml` for API, worker, frontend, PostgreSQL/pgvector, Redis, MinIO, Prometheus, Grafana, and OTel Collector. |

## Architecture Diagram

![Architecture Diagram](public/Architecture%20Diagram.png)

## Repository Layout

```text
frontend/                  Next.js chat UI and API client
src/memos_q/api.py          FastAPI service and endpoint definitions
src/memos_q/engine.py       MemoryOS orchestration and maintenance logic
src/memos_q/store.py        JSON/in-memory/Postgres-compatible memory storage
src/memos_q/integrations/   Qwen, storage, Pinecone, Redis, and S3 adapters
src/memos_q/workers/        Celery app and background tasks
src/memos_q/monitoring/     Prometheus, OpenTelemetry, and Langfuse helpers
monitoring/                 Prometheus, Grafana, and OTel Collector configs
tests/                      Unit/API tests for memory, embeddings, and FastAPI
```

## Prerequisites

- Python 3.10+
- Node.js 20+ and npm for the frontend
- Docker Engine with Docker Compose v2 for the full local stack
- Optional Qwen/DashScope API key for live model calls
- Optional Pinecone, Langfuse, and object-storage credentials for production-like deployments

## Configuration

The backend loads environment variables directly and also reads a repository-root `.env` file when present. Create one manually for Docker Compose or local runs:

```bash
cat > .env <<'ENV'
MEMOS_ENV=development
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Local defaults
MEMOS_STORE=json
MEMOS_JSON_PATH=.memos/memory-store.json
POSTGRES_USER=memos
POSTGRES_PASSWORD=memos
POSTGRES_DB=memos
POSTGRES_DSN=postgresql://memos:memos@postgres:5432/memos
REDIS_URL=redis://:ecs-2030@redis:6379/0
CELERY_BROKER_URL=redis://:ecs-2030@redis:6379/1
CELERY_RESULT_BACKEND=redis://:ecs-2030@redis:6379/2
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY_ID=memos
S3_SECRET_ACCESS_KEY=memos-password
S3_BUCKET=memos-q
MINIO_ROOT_USER=memos
MINIO_ROOT_PASSWORD=memos-password

# Live Qwen/DashScope settings
QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_DEFAULT_MODEL=qwen3.5-flash
QWEN_REASONING_MODEL=qwen3.5-plus
QWEN_FLASH_MODEL=qwen3.5-flash
QWEN_VL_MODEL=qwen3-vl-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_EMBEDDING_DIMENSIONS=1024
QWEN_REQUIRE_LIVE_EMBEDDINGS=false

# Optional production services
PINECONE_API_KEY=
PINECONE_HOST=
PINECONE_INDEX=memos-q-vectors
PINECONE_NAMESPACE=memos-q
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
MEMOS_ENABLE_OTEL=false
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
ENV
```

Storage modes:

- `MEMOS_STORE=json` — default local development mode using `.memos/memory-store.json`.
- `MEMOS_STORE=memory` — ephemeral mode for tests and quick experiments.
- `MEMOS_STORE=postgres` — PostgreSQL-backed memory records.
- `MEMOS_STORE=alicloud` — production-oriented mode using PostgreSQL records with managed/external vector recall and Alibaba-compatible services.

Set `QWEN_REQUIRE_LIVE_EMBEDDINGS=true` only when the Qwen embedding service must be available; otherwise the app can use deterministic fallback embeddings for development and tests.

## Run the Full Stack with Docker Compose

```bash
docker compose up --build
```

Open:

- Frontend chat: <http://localhost:3000>
- FastAPI docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3001>
- MinIO console: <http://localhost:9001>

Useful Docker commands:

```bash
docker compose ps
docker compose logs -f api worker frontend
docker compose up --build api
docker compose up --build frontend
docker compose down
docker compose down -v
```

## Run Locally Without Docker

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api,live,monitoring,test]'
MEMOS_STORE=json uvicorn memos_q.api:app --host 0.0.0.0 --port 8000 --reload
```

For an ephemeral backend:

```bash
MEMOS_STORE=memory uvicorn memos_q.api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

Then open <http://localhost:3000>.

### Worker

```bash
docker compose up -d redis
source .venv/bin/activate
CELERY_BROKER_URL=redis://:ecs-2030@localhost:6379/1 \
CELERY_RESULT_BACKEND=redis://:ecs-2030@localhost:6379/2 \
celery -A memos_q.workers.celery_app.celery_app worker --loglevel=INFO
```

## API Smoke Tests

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
  -d '{"query": "How should I answer this user about agents?", "limit": 3}'
```

```bash
curl -N -X POST http://localhost:8000/agent/chat \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: demo-user' \
  -d '{"message": "What do you remember about me?", "source_session": "smoke-test"}'
```

```bash
curl http://localhost:8000/integrations/status
```

## Important Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health. |
| `POST /memories` | Create a memory for the authenticated `x-user-id`. |
| `POST /recall` | Retrieve relevant memories with explainable scores. |
| `POST /agent/chat` | Stream a Qwen-backed memory-aware chat response. |
| `POST /agent/qwen-agent` | Run the optional Qwen-Agent integration. |
| `POST /ingest/vision` | Extract memory candidates from an image/document URL with Qwen3-VL. |
| `GET /users/me/memories` | Inspect active or inactive memories. |
| `POST /users/me/memories/{memory_id}/approve` | Approve a pending memory. |
| `POST /users/me/memories/{memory_id}/reject` | Reject a pending memory. |
| `PATCH /users/me/memories/{memory_id}` | Edit a memory. |
| `POST /users/me/memories/{memory_id}/archive` | Archive a memory. |
| `DELETE /users/me/memories/{memory_id}` | Delete a memory. |
| `POST /users/me/maintenance` | Run maintenance for the current user. |
| `POST /admin/reconcile-vectors` | Reconcile vector storage for the current user. |
| `GET /integrations/status` | Return frontend-readable integration configuration status. |
| `GET /metrics` | Prometheus metrics. |

## Testing and Validation

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
cd frontend
npm run build
```

## Troubleshooting

- If `docker compose` reports missing variables, create the `.env` file from the template in this README.
- If the frontend cannot reach the backend, confirm `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` and that `curl http://localhost:8000/health` returns `{"status":"ok"}`.
- If live chat fails with a Qwen authentication error, set `QWEN_API_KEY` and restart the API.
- If Docker-backed Redis commands fail locally, include the password from the compose command: `redis://:ecs-2030@localhost:6379/<db>`.
- If you do not want external services during development, use `MEMOS_STORE=json` or `MEMOS_STORE=memory` and leave `QWEN_REQUIRE_LIVE_EMBEDDINGS=false`.
- If frontend dependencies fail to install, check npm registry/auth configuration and rerun `npm install` inside `frontend/`.

## Built for Qwen Workflows

MemOS-Q demonstrates how QwenCloud-backed agents can remember responsibly across sessions while keeping recall auditable, configurable, and production-ready.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.