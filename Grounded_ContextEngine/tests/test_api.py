from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_rag_service
from app.main import app
from app.schemas import DocumentSource, IngestResponse, QueryResponse


class FakeRAGService:
    async def ingest(self, tenant_id, filename, content):
        assert tenant_id == "demo"
        assert "service level" in content
        return IngestResponse(document_id=uuid4(), filename=filename, chunks_created=1)

    async def query(self, tenant_id, query, top_k):
        return QueryResponse(
            answer="[1] The service level is one hour.",
            sources=[
                DocumentSource(
                    document_id=uuid4(),
                    filename="policy.txt",
                    chunk_index=0,
                    content="The service level is one hour.",
                    score=0.9,
                )
            ],
        )

    async def delete_document(self, tenant_id, document_id):
        return True

    async def stream(self, tenant_id, query, top_k):
        yield "event: done\ndata: {}\n\n"


@pytest_asyncio.fixture
async def client():
    app.dependency_overrides[get_rag_service] = FakeRAGService
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_public_dashboard_and_pipeline_overview(client):
    page = await client.get("/")
    assert page.status_code == 200
    assert "Pipeline observability" in page.text
    assert "runDemo" in page.text

    overview = await client.get("/api/v1/observability/overview")
    assert overview.status_code == 200
    data = overview.json()
    assert {pipeline["id"] for pipeline in data["pipelines"]} == {
        "ingestion", "retrieval", "generation"
    }
    assert "tenant_id" not in overview.text


@pytest.mark.asyncio
async def test_public_sample_policy_is_served(client):
    response = await client.get("/demo/sample_policy.txt")
    assert response.status_code == 200
    assert "one-hour initial response" in response.text


@pytest.mark.asyncio
async def test_document_ingestion_and_query(client):
    headers = {"X-Tenant-ID": "demo"}
    ingested = await client.post(
        "/api/v1/rag/documents",
        headers=headers,
        files={"file": ("policy.txt", b"The service level is one hour.", "text/plain")},
    )
    assert ingested.status_code == 201
    assert ingested.json()["chunks_created"] == 1

    response = await client.post(
        "/api/v1/rag/query", headers=headers, json={"query": "What is the service level?"}
    )
    assert response.status_code == 200
    assert "one hour" in response.json()["answer"]
    assert response.json()["sources"][0]["filename"] == "policy.txt"


@pytest.mark.asyncio
async def test_unsupported_document_type_is_rejected(client):
    response = await client.post(
        "/api/v1/rag/documents",
        headers={"X-Tenant-ID": "demo"},
        files={"file": ("image.png", b"not a document", "image/png")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_stream_endpoint_emits_server_sent_events(client):
    response = await client.post(
        "/api/v1/rag/stream",
        headers={"X-Tenant-ID": "demo"},
        json={"query": "What is the service level?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
