import io
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pypdf import PdfReader

from app.api.deps import get_rag_service, get_tenant_id
from app.core.config import Settings, get_settings
from app.schemas import IngestResponse, QueryRequest, QueryResponse
from app.services.rag_service import RAGService
from app.telemetry import telemetry

logger = logging.getLogger(__name__)
router = APIRouter()


def extract_text(filename: str, payload: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in {"txt", "md", "markdown", "csv", "log"}:
        return payload.decode("utf-8-sig")
    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise HTTPException(status_code=415, detail="Supported file types: txt, md, csv, log, pdf")


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    rag: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    async with telemetry.track("upload"):
        payload = await file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the configured upload limit")
    filename = file.filename or "upload.txt"
    try:
        async with telemetry.track("extract"):
            content = extract_text(filename, payload)
        result = await rag.ingest(tenant_id, filename, content)
    except (UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    logger.info(
        "document_ingested",
        extra={"tenant_id": tenant_id, "document_id": str(result.document_id), "filename": filename},
    )
    return result


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
    rag: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    return await rag.query(tenant_id, payload.query, payload.top_k or settings.top_k_default)


@router.post("/stream")
async def stream_query(
    payload: QueryRequest,
    tenant_id: str = Depends(get_tenant_id),
    rag: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    async def events():
        async for event in rag.stream(tenant_id, payload.query, payload.top_k or settings.top_k_default):
            yield event

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    tenant_id: str = Depends(get_tenant_id),
    rag: RAGService = Depends(get_rag_service),
) -> None:
    if not await rag.delete_document(tenant_id, document_id):
        raise HTTPException(status_code=404, detail="Document not found")
