# MemOS-Q: A Self-Evolving Memory Operating System for AI Agents

Built with QwenCloud, **MemOS-Q** is a production-oriented memory layer that enables AI agents to remember, reason, adapt, and self-correct across conversations, documents, images, and external tools.

Unlike stateless chatbots that rely only on context windows, MemOS-Q introduces persistent memory with confidence scoring, conflict resolution, multimodal understanding, explainable recall, and autonomous memory maintenance.

## Why MemOS-Q?

Modern AI assistants commonly struggle with:

1. **No long-term memory** — information is forgotten between sessions and users repeat the same context.
2. **Memory hallucinations** — incorrect or outdated facts can be stored with little transparency.
3. **Lack of multimodal continuity** — facts from screenshots, PDFs, slide decks, repositories, and documents are disconnected from later conversations.
4. **Poor auditability** — users cannot easily inspect why something was remembered or forgotten.

MemOS-Q treats memory management as infrastructure for autonomous AI systems.

## Core Capabilities

### Explainable Memory

Every recalled memory includes:

- source session
- confidence score
- timestamp
- ranking signals
- reasoning path

### Self-Correcting Memory

The memory quality engine detects contradictions, outdated information, and superseded preferences. Newer high-confidence memories can deactivate older conflicting memories while preserving an audit trail.

### Multimodal Memory

Using Qwen3-VL, MemOS-Q can create memories from PDFs, screenshots, slide decks, diagrams, images, and whiteboard photos.

### Autonomous Memory Maintenance

Maintenance jobs continuously merge duplicates, summarize old conversations, promote stable facts, archive stale information, and update confidence scores.

## Architecture

```text
User Input
   ↓
Orchestrator Agent
   ↓
Memory Pipeline
   ├── Retrieval Agent
   ├── Memory Agent
   ├── Profile Agent
   ├── Audit Agent
   └── Compaction Agent
   ↓
Qwen Models
   ↓
Response Generation
   ↓
Memory Write-Back
```

## QwenCloud Usage

- **Qwen3.5-Plus** — response synthesis, long-context reasoning, memory decisions, profile inference, and multi-turn dialogue.
- **Qwen3.5-Flash** — intent classification, memory extraction, memory scoring, deduplication, routing, and summarization.
- **Qwen3-VL-Plus** — screenshot understanding, PDF extraction, slide analysis, image memory creation, and document summarization.
- **Qwen Batch API** — nightly compaction, clustering, cleanup, and profile consolidation.
- **Qwen-Agent** — tool calling, memory workflows, MCP integrations, and multi-agent coordination.

## Memory Layers

| Layer | Purpose | Example |
| --- | --- | --- |
| Working memory | Single-session context | Active task state |
| Episodic memory | Session-based events | User uploaded a system design document |
| Semantic memory | Stable long-term facts | User uses Python |
| Operational memory | Agent execution knowledge | Failed tool calls and safety decisions |

## Memory Graph

MemOS-Q stores relationship edges alongside vector-style memory content.

**Node types:** preference, fact, project, event, task.

**Relationship types:** derived_from, contradicts, supersedes, related_to, supports.

This graph improves retrieval quality and makes recall explainable.

## Prototype Implementation

This repository includes a lightweight Python implementation of the core memory primitives:

- in-memory storage adapter
- scoring engine for importance, confidence, novelty, and stability
- hybrid retrieval with explainable ranking signals
- contradiction and supersession handling
- duplicate merging and confidence decay maintenance
- optional FastAPI app factory for service deployment

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[api,test]
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

## User Control

Supported memory actions include:

- remember this
- forget this
- update memory
- inspect memory
- view memory graph

All mutating changes are logged in the audit trail.

## Future Work

- Cross-agent shared memory
- Federated memory synchronization
- Team knowledge graphs
- Long-term behavioral adaptation
- Agent-to-agent memory exchange
- Reinforcement-based memory ranking

## Built for Qwen Code Challenge

MemOS-Q demonstrates QwenCloud-powered memory workflows while addressing a fundamental challenge for next-generation AI systems: **How can AI remember responsibly, transparently, and at scale?**
