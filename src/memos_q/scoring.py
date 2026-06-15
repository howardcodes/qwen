"""Scoring helpers for memory quality and retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import datetime

from .models import Memory, clamp_score, utc_now

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Tokenize text for deterministic local retrieval."""

    return set(TOKEN_RE.findall(text.lower()))




def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    """Return cosine similarity mapped to the inclusive [0, 1] range."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return clamp_score((dot + 1.0) / 2.0)


def keyword_score(query: str, memory: Memory) -> float:
    """Return token overlap between a query and a memory."""

    query_tokens = tokenize(query)
    memory_tokens = tokenize(memory.content) | memory.tags
    if not query_tokens or not memory_tokens:
        return 0.0
    return len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)


def semantic_score(query: str, memory: Memory, query_embedding: list[float] | None = None) -> float:
    """Return Qwen embedding cosine similarity with keyword explainability only.

    Query embeddings must be supplied by the caller so production recall cannot
    silently downgrade to local or hardcoded embeddings.
    """

    vector_score = cosine_similarity(query_embedding, memory.embedding)
    query_tokens = tokenize(query)
    memory_tokens = tokenize(memory.content)
    token_score = 0.0 if not query_tokens or not memory_tokens else len(query_tokens & memory_tokens) / math.sqrt(len(query_tokens) * len(memory_tokens))
    return max(vector_score, token_score)


def recency_score(timestamp: datetime | None) -> float:
    """Score recent memories higher with a smooth time decay."""

    if timestamp is None:
        return 0.5
    age_days = max(0.0, (utc_now() - timestamp).total_seconds() / 86_400)
    return clamp_score(1 / (1 + age_days / 30))


def graph_proximity_score(memory: Memory, query_tags: Iterable[str]) -> float:
    """Score memories sharing tags with the query context."""

    tags = {tag.lower() for tag in query_tags}
    if not tags or not memory.tags:
        return 0.0
    return len(tags & memory.tags) / len(tags | memory.tags)


def hybrid_retrieval_score(
    query: str,
    memory: Memory,
    *,
    query_tags: Iterable[str] = (),
    query_embedding: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute weighted vector/keyword retrieval score and individual signals."""

    signals = {
        "vector": cosine_similarity(query_embedding, memory.embedding),
        "semantic": semantic_score(query, memory, query_embedding=query_embedding),
        "keyword": keyword_score(query, memory),
        "recency": recency_score(memory.last_recalled_at or memory.updated_at),
        "importance": memory.importance_score,
        "confidence": memory.confidence_score,
        "graph": graph_proximity_score(memory, query_tags),
    }
    score = (
        0.30 * signals["vector"]
        + 0.20 * signals["semantic"]
        + 0.15 * signals["keyword"]
        + 0.10 * signals["recency"]
        + 0.10 * signals["importance"]
        + 0.10 * signals["confidence"]
        + 0.05 * signals["graph"]
    )
    return clamp_score(score), signals


def quality_scores(content: str, existing_memories: Iterable[Memory]) -> dict[str, object]:
    """Estimate quality scores and reasons for a candidate memory."""

    words = tokenize(content)
    matching_existing = [keyword_score(content, memory) for memory in existing_memories]
    max_overlap = max(matching_existing, default=0.0)
    stable_markers = {"prefer", "prefers", "uses", "works", "interested", "always", "usually"}
    reasons = ["deterministic local quality scorer"]
    if words & stable_markers:
        reasons.append("contains stable preference/fact markers")
    if max_overlap > 0.7:
        reasons.append("similar to an existing memory")
    return {
        "importance": clamp_score(min(1.0, 0.35 + len(words) / 40)),
        "confidence": 0.8,
        "novelty": clamp_score(1 - max_overlap),
        "stability": clamp_score(0.45 + (0.25 if words & stable_markers else 0.0)),
        "confidence_reasons": reasons,
    }
