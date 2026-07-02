# MemOS-Q: A Self-Evolving Persistent Memory Agentic AI System

> **Track 1: MemoryAgent Submission**

MemOS-Q is a persistent-memory agentic AI system that transforms stateless large language models into continuously learning autonomous agents. Built around a self-evolving Memory Operating System (MemoryOS), MemOS-Q combines persistent memory, agentic planning, autonomous reflection, proactive task management, multimodal understanding, and long-term personalization.

Unlike traditional AI assistants that lose context between sessions, MemOS-Q continuously accumulates experiences, learns user preferences, recalls relevant knowledge, forgets outdated information, and autonomously performs actions on behalf of users.

---

# Overview

MemOS-Q combines:

* **Persistent cross-session memory**
* **Agentic planning and reasoning**
* **Autonomous experience accumulation**
* **Long-term user preference learning**
* **Explainable memory retrieval**
* **Memory evolution and forgetting**
* **Proactive daily briefings**
* **Telegram agent notifications**
* **Multimodal memory ingestion**
* **Production observability**
* **Cloud-native deployment**

The platform consists of:

* A **Next.js frontend** chat interface
* A **FastAPI backend**
* A **MemoryOS engine**
* An **agentic orchestration layer**
* **Qwen/Qwen-Agent integrations**
* **Celery background workers**
* **Telegram notification agents**
* **Production observability tooling**

---

# Key Features

## 🧠 Persistent Cross-Session Memory

MemOS-Q maintains durable memory across conversations and sessions.

Examples include:

* User preferences
* Historical interactions
* Long-term goals
* Personal profiles
* Behavioral patterns
* Session summaries
* Open tasks
* User habits

---

## 🤖 Agentic AI Architecture

MemOS-Q extends traditional memory agents into a fully agentic system.

The agent autonomously:

* Plans actions
* Retrieves relevant memories
* Executes tools
* Reflects on outcomes
* Learns from experience
* Updates user profiles
* Maintains long-term goals
* Generates proactive recommendations

Core components:

* Planner
* Router
* Tool Executor
* Reflection Engine
* Safety Layer
* Notification Engine
* Agent State Manager

---

## 🔄 Autonomous Experience Accumulation

The system automatically extracts and stores durable information.

Examples:

* Travel plans
* Career goals
* Educational history
* User preferences
* Personal interests
* Frequently discussed topics
* Daily activities
* Long-term objectives

---

## 🎯 Intelligent Memory Recall

MemOS-Q retrieves only the most relevant memories using hybrid retrieval.

Features:

* Semantic search
* Vector similarity retrieval
* Relevance scoring
* Explainable recall
* Memory ranking
* Context optimization
* Conflict detection

---

## 🧩 Memory Evolution

Memories continuously evolve through:

* Summarization
* Consolidation
* Deduplication
* Conflict resolution
* User profile updates
* Memory merging
* Knowledge refinement

---

## 🗑️ Timely Forgetting

The memory maintenance system prevents memory pollution through:

* Time decay
* Importance scoring
* Duplicate detection
* Memory compaction
* Archiving
* Summarization
* Conflict resolution

---

## 👤 User Preference Learning

MemOS-Q continuously learns:

* Communication preferences
* Personal interests
* Travel preferences
* Food preferences
* Behavioral tendencies
* Work habits
* Interaction styles

---

## 📋 Persistent Task Memory

The agent automatically extracts and manages durable tasks.

Task metadata includes:

* Task title
* Status
* Blockers
* Next actions
* Confidence
* Evidence
* Completion state

Supported states:

* Open
* In Progress
* Blocked
* Done
* Dropped

---

## 🌅 Proactive Daily Briefings

MemOS-Q autonomously generates daily summaries containing:

* Open tasks
* User priorities
* Recent memories
* Upcoming actions
* Suggested next steps
* Personalized recommendations

---

## 📲 Telegram Agent Notifications

MemOS-Q can proactively communicate via Telegram.

Examples:

* Daily summaries
* Goal reminders
* Task follow-ups
* Agent recommendations
* System notifications
* Memory insights

---

## 🖼️ Multimodal Memory Ingestion

The system supports memory extraction from:

* Text
* Images
* Documents
* Visual understanding
* Qwen-VL processing

---

## 👁️ Explainable Memory Recall

Every retrieved memory contains:

* Relevance score
* Confidence score
* Retrieval explanation
* Source information
* Ranking position
* Importance score

---

## 📊 Production Observability

Built-in observability includes:

* Prometheus metrics
* Grafana dashboards
* OpenTelemetry tracing
* Langfuse integration
* Agent telemetry
* Memory analytics
* Health monitoring
* Performance analytics

---

# Agentic Workflow

MemOS-Q executes an agentic pipeline consisting of:

```text
User Input
      ↓
Load Context
      ↓
Retrieve Memories
      ↓
Load Tasks
      ↓
Check System Health
      ↓
Planner
      ↓
Tool Execution
      ↓
Reflection
      ↓
Memory Update
      ↓
Notification Decision
      ↓
Persist Results
```

The agent continuously learns from every execution cycle.

---

# Current Architecture

![Architecture](/public/Architecture%20Diagram.png)

---

# Repository Structure

```text
frontend/
    components/
    pages/
    lib/

src/memos_q/
    agentic/
        graph.py
        planner.py
        router.py
        reflection.py
        tools.py
        memory.py
        notification.py
        safety.py
        state.py
        prompts.py

    api.py
    engine.py
    store.py
    daily_summary.py
    telegram.py
    scoring.py

    integrations/
        qwen_cloud.py
        durable.py
        factory.py

    monitoring/
        agent_metrics.py
        memory_metrics.py
        observability.py

    workers/
        celery_app.py

monitoring/
    grafana/
    prometheus/
    otel-collector/

tests/
```

---

# Storage Modes

MemOS-Q supports multiple storage backends:

| Mode     | Description              |
| -------- | ------------------------ |
| memory   | In-memory testing        |
| json     | Local persistent storage |
| postgres | PostgreSQL-backed memory |
| alicloud | Cloud deployment mode    |

---

# Docker Architecture

```text
Frontend (3000)
        ↓
FastAPI API (8000)
        ↓
MemoryOS Engine
        ↓
Agentic Workflow
        ↓
Qwen Cloud
        ↓
PostgreSQL
Redis
MinIO
Pinecone
Telegram
```

Additional services:

* Celery Worker
* Celery Beat
* Prometheus
* Grafana
* OpenTelemetry Collector

---

# Running Locally

## Start entire stack

```bash
docker compose up --build
```

Services:

| Service    | URL                        |
| ---------- | -------------------------- |
| Frontend   | http://localhost:3000      |
| API        | http://localhost:8000      |
| Swagger    | http://localhost:8000/docs |
| Prometheus | http://localhost:9090      |
| Grafana    | http://localhost:3001      |
| MinIO      | http://localhost:9001      |

---

# Environment Variables

Important configuration:

```bash
# MemOS-Q production ECS / Alibaba Cloud configuration
MEMOS_ENV=production
API_BASE_URL=http://47.236.145.69:8000
NEXT_PUBLIC_API_BASE_URL=http://47.236.145.69:8000
FRONTEND_URL=http://47.236.145.69:3000

# Production mode: ECS runs FastAPI plus Postgres/Redis/MinIO, Qwen/DashScope
# creates embeddings, and Pinecone stores/searches vectors.
MEMOS_STORE=alicloud
QWEN_REQUIRE_LIVE_EMBEDDINGS=true

# Qwen / DashScope Model Studio
QWEN_API_KEY=qwen_api_key_here
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_REASONING_MODEL=qwen3.5-plus
QWEN_FLASH_MODEL=qwen3.5-flash
QWEN_VL_MODEL=qwen3-vl-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_EMBEDDING_DIMENSIONS=1024
QWEN_CHAT_DEFAULT_MODEL=qwen3.5-flash
QWEN_CHAT_MAX_TOKENS=800
QWEN_REASONING_MAX_TOKENS=1200
QWEN_MEMORY_EXTRACTION_MAX_TOKENS=500
QWEN_CONFLICT_RESOLUTION_MAX_TOKENS=300
QWEN_SUMMARY_MAX_TOKENS=400
MEMORY_RECALL_TOP_K=5
MEMORY_RECALL_VECTOR_TOP_K=10
MEMORY_RECALL_FALLBACK_LIMIT=20
MEMORY_EXTRACTION_INCLUDE_ASSISTANT_RESPONSE=false
MEMORY_EXTRACTION_MAX_INPUT_CHARS=4000
MEMORY_AUTO_APPROVE_CONFIDENCE=0.80
MEMORY_CONFLICT_CONFIRMATION_ENABLED=true

# Postgres on ECS source of truth for memory/audit records
POSTGRES_USER=memos
POSTGRES_PASSWORD=postgres_password_here
POSTGRES_DB=memos
POSTGRES_DSN=postgresql://memos:postgres_password_here@postgres:5432/memos

# Pinecone vector database for Qwen/DashScope embeddings
PINECONE_API_KEY=pinecone_api_key_here
PINECONE_HOST=pinecone_host_here
PINECONE_INDEX=memos-q-vectors
PINECONE_NAMESPACE=memos-q

# Redis / Celery on ECS
REDIS_URL=redis://:redis_password_here@redis:6379/0
CELERY_BROKER_URL=redis://:celery_password_here@redis:6379/1
CELERY_RESULT_BACKEND=redis://:celery_password_here@redis:6379/2

# MinIO on ECS (S3-compatible endpoint)
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY_ID=admin
S3_SECRET_ACCESS_KEY=s3_secret_access_key_here
S3_BUCKET=memos-q
S3_REGION=us-east-1

# observability (LLMOps)
LANGFUSE_PUBLIC_KEY=langfuse_public_key_here
LANGFUSE_SECRET_KEY=langfuse_secret_key_here
LANGFUSE_HOST=https://us.cloud.langfuse.com
MEMOS_ENABLE_OTEL=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

# Telegram API
TELEGRAM_CHAT_ID=telegram_chat_id_here
TELEGRAM_BOT_TOKEN=telegram_bot_token_here
```

---

# Major API Endpoints

| Endpoint                      | Description            |
| ----------------------------- | ---------------------- |
| GET /health                   | Health check           |
| POST /memories                | Create memory          |
| POST /recall                  | Retrieve memories      |
| POST /agent/chat              | Agentic chat           |
| POST /agent/qwen-agent        | Qwen-Agent execution   |
| POST /ingest/vision           | Vision ingestion       |
| POST /users/me/maintenance    | Memory maintenance     |
| GET /users/me/memories        | Retrieve user memories |
| POST /admin/reconcile-vectors | Vector reconciliation  |
| GET /metrics                  | Prometheus metrics     |
| GET /integrations/status      | Integration status     |

---

# Testing

```bash
pytest -q
```

```bash
python -m compileall src tests
```

```bash
cd frontend
npm run build
```

---

# Demonstrating Persistent Memory

### Session 1

> "I'm going to North Carolina for exchange next semester."

MemOS-Q stores:

* Location
* Event
* Time context
* User profile updates

---

### Session 8

> "What nearby cities should I visit?"

Traditional LLM:

> "Which city are you staying in?"

MemOS-Q:

> "Since you'll be staying in North Carolina for your exchange, you may enjoy Asheville, Charleston, and the Blue Ridge Parkway."

This demonstrates:

* Persistent memory
* Cross-session recall
* User preference learning
* Experience accumulation
* Agentic reasoning
* Long-term personalization

---


# License

MIT License.
