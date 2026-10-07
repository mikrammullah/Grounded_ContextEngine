import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from threading import Lock
from typing import Any


PIPELINES = [
    {
        "id": "ingestion",
        "name": "Document ingestion",
        "description": "Upload, extract, chunk, embed, and index source documents.",
        "stages": ["upload", "extract", "chunk", "embed", "index"],
    },
    {
        "id": "retrieval",
        "name": "Grounded retrieval",
        "description": "Embed a question and retrieve tenant-scoped context from pgvector.",
        "stages": ["query_embed", "retrieve"],
    },
    {
        "id": "generation",
        "name": "Answer generation",
        "description": "Generate a context-grounded answer with source citations.",
        "stages": ["generate", "stream"],
    },
]

STAGE_LABELS = {
    "upload": "Receive upload",
    "extract": "Extract text",
    "chunk": "Split into chunks",
    "embed": "Create embeddings",
    "index": "Write to pgvector",
    "query_embed": "Embed question",
    "retrieve": "Search tenant index",
    "generate": "Generate answer",
    "stream": "Stream answer tokens",
}


class TelemetryLedger:
    """Bounded, process-local activity ledger for operational visibility."""

    def __init__(self, max_events: int = 300):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._stages: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self._next_id = 1

    def record(self, stage: str, status: str, duration_ms: float | None = None) -> dict[str, Any]:
        pipeline_id = next(
            pipeline["id"] for pipeline in PIPELINES if stage in pipeline["stages"]
        )
        now = datetime.now(UTC).isoformat()
        with self._lock:
            event = {
                "id": self._next_id,
                "timestamp": now,
                "pipeline_id": pipeline_id,
                "pipeline": next(item["name"] for item in PIPELINES if item["id"] == pipeline_id),
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
                "status": status,
                "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            }
            self._next_id += 1
            self._events.append(event)
            current = self._stages.setdefault(stage, {"runs": 0})
            current.update({"status": status, "timestamp": now, "duration_ms": event["duration_ms"]})
            if status in {"completed", "failed"}:
                current["runs"] += 1
            return event

    def overview(self) -> dict[str, Any]:
        with self._lock:
            pipelines = []
            for pipeline in PIPELINES:
                stages = []
                for stage in pipeline["stages"]:
                    last = self._stages.get(stage, {})
                    status = last.get("status", "idle")
                    stages.append(
                        {
                            "id": stage,
                            "name": STAGE_LABELS[stage],
                            "status": status,
                            "last_seen": last.get("timestamp"),
                            "duration_ms": last.get("duration_ms"),
                            "runs": last.get("runs", 0),
                        }
                    )
                states = {stage["status"] for stage in stages}
                pipeline_status = (
                    "degraded" if "failed" in states
                    else "running" if "running" in states
                    else "active" if "completed" in states
                    else "idle"
                )
                pipelines.append({**pipeline, "status": pipeline_status, "stages": stages})
            return {
                "service": "Enterprise RAG Service",
                "updated_at": datetime.now(UTC).isoformat(),
                "retained_events": len(self._events),
                "pipelines": pipelines,
            }

    def events_after(self, event_id: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [event.copy() for event in self._events if event["id"] > event_id][-limit:]

    @asynccontextmanager
    async def track(self, stage: str) -> AsyncIterator[None]:
        started_at = time.perf_counter()
        self.record(stage, "running")
        try:
            yield
        except Exception:
            self.record(stage, "failed", (time.perf_counter() - started_at) * 1000)
            raise
        else:
            self.record(stage, "completed", (time.perf_counter() - started_at) * 1000)


telemetry = TelemetryLedger()
