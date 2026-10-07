import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.responses import FileResponse, Response

from app.api.observability import router as observability_router
from app.api.rag import router as rag_router
from app.core.config import get_settings
from app.core.database import engine, lifespan_session

settings = get_settings()
REQUESTS = Counter("rag_http_requests_total", "HTTP requests handled", ["method", "path"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level.upper())
    async with lifespan_session():
        yield


app = FastAPI(title=settings.project_name, version=settings.version, lifespan=lifespan)
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(observability_router, prefix="/api/v1/observability", tags=["Observability"])


@app.middleware("http")
async def count_requests(request, call_next):
    response = await call_next(request)
    REQUESTS.labels(request.method, request.url.path).inc()
    return response


@app.get("/healthz", tags=["Health"])
async def healthz():
    return {"status": "healthy", "version": settings.version}


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/demo/sample_policy.txt", include_in_schema=False)
async def sample_policy():
    return FileResponse(Path(__file__).parent / "static" / "sample_policy.txt", media_type="text/plain")


@app.get("/readyz", tags=["Health"])
async def readyz():
    async with engine.connect() as connection:
        await connection.exec_driver_sql("SELECT 1")
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
