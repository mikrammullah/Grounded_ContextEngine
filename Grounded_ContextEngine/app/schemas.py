from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    chunks_created: int


class DocumentSource(BaseModel):
    document_id: UUID
    filename: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[DocumentSource]
