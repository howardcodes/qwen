"""Monitoring hooks for Langfuse, OpenTelemetry, Prometheus, and Grafana."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memos_q.config import Settings, settings


def configure_opentelemetry(app: Any, config: Settings = settings) -> None:
    """Instrument FastAPI with OpenTelemetry OTLP export."""

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "memos-q-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otel_exporter_otlp_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


def add_prometheus_metrics(app: Any) -> None:
    """Expose Prometheus metrics for Grafana dashboards."""

    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def langfuse_trace(name: str, config: Settings = settings) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function with a Langfuse trace when credentials are set."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                return func(*args, **kwargs)
            from langfuse import Langfuse

            client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
            )
            trace = client.trace(name=name)
            try:
                result = func(*args, **kwargs)
                trace.update(output=result)
                return result
            finally:
                client.flush()

        return wrapper

    return decorator
