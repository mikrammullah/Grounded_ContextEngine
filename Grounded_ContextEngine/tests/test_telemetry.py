import pytest

from app.telemetry import TelemetryLedger


@pytest.mark.asyncio
async def test_track_records_running_and_completed_stage():
    ledger = TelemetryLedger()
    async with ledger.track("retrieve"):
        assert ledger.overview()["pipelines"][1]["stages"][1]["status"] == "running"

    events = ledger.events_after()
    assert [event["status"] for event in events] == ["running", "completed"]
    assert ledger.overview()["pipelines"][1]["stages"][1]["runs"] == 1


@pytest.mark.asyncio
async def test_track_records_failed_stage_and_reraises():
    ledger = TelemetryLedger()
    with pytest.raises(RuntimeError):
        async with ledger.track("generate"):
            raise RuntimeError("model unavailable")

    assert ledger.events_after()[-1]["status"] == "failed"
    assert ledger.overview()["pipelines"][2]["status"] == "degraded"