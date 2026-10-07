from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.rag_service import RAGService


class RecordingSession:
    def __init__(self):
        self.rows = []

    def add_all(self, rows):
        self.rows = rows

    async def commit(self):
        pass


class DeterministicModels:
    async def embed(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


@pytest.mark.asyncio
async def test_ingest_indexes_child_text_and_retains_parent_context():
    session = RecordingSession()
    service = RAGService(
        session,
        Settings(
            chunk_size=100,
            chunk_overlap=10,
            child_chunk_size=40,
            child_chunk_overlap=5,
        ),
    )
    service.models = DeterministicModels()
    content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 5

    response = await service.ingest("tenant-a", "legal.txt", content)

    assert response.chunks_created == len(session.rows)
    assert len(session.rows) > 1
    assert all(row.content in row.metadata_json["_rag_parent_text"] for row in session.rows)
    assert all(row.tenant_id == "tenant-a" for row in session.rows)

    row = session.rows[0]
    source = RAGService._sources([(row, 0.8)])[0]
    assert source.content == row.metadata_json["_rag_parent_text"]
    assert "_rag_parent_text" not in source.metadata
    assert "_rag_parent_index" not in source.metadata


def test_context_text_falls_back_to_legacy_chunk_content():
    chunk = SimpleNamespace(metadata_json={}, content="legacy child content")

    assert RAGService._context_text(chunk) == "legacy child content"