import asyncio
import json
import logging

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse

from app.telemetry import telemetry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/overview")
async def pipeline_overview():
    return telemetry.overview()


@router.get("/events")
async def pipeline_events(
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
):
    return {"events": telemetry.events_after(after, limit)}


@router.get("/stream")
async def stream_pipeline_events(
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    try:
        cursor = max(0, int(last_event_id or 0))
    except ValueError:
        cursor = 0

    async def events():
        nonlocal cursor
        while not await request.is_disconnected():
            pending = telemetry.events_after(cursor, 100)
            if pending:
                for event in pending:
                    cursor = event["id"]
                    yield f"id: {cursor}\nevent: pipeline\ndata: {json.dumps(event)}\n\n"
            else:
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
