"""Prometheus metrics for the memory lifecycle pipeline."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

memories_created_total = Counter("memories_created_total", "Memories created")
memories_approved_total = Counter("memories_approved_total", "Memories approved")
memories_rejected_total = Counter("memories_rejected_total", "Memories rejected")
memories_deleted_total = Counter("memories_deleted_total", "Memories forgotten/deleted")
memories_recalled_total = Counter("memories_recalled_total", "Memories recalled")
conflict_count = Counter("conflict_count", "Memory conflicts detected")
duplicate_count = Counter("duplicate_count", "Memory duplicates detected")
qwen_errors_total = Counter("qwen_errors_total", "Qwen errors")

active_memory_count = Gauge("active_memory_count", "Active memories")
pending_review_count = Gauge("pending_review_count", "Pending review memories")
conflicting_memory_count = Gauge("conflicting_memory_count", "Conflicting memories")

memory_recall_score = Histogram("memory_recall_score", "Memory recall score")
qwen_latency_seconds = Histogram("qwen_latency_seconds", "Qwen latency")
memory_search_latency_seconds = Histogram("memory_search_latency_seconds", "Memory search latency")
memory_extraction_latency_seconds = Histogram("memory_extraction_latency_seconds", "Memory extraction latency")
