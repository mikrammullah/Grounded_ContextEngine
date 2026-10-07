# Grounded AI Context Engine
(bringing relevant info to AI)

A tenant-aware document intelligence platform
An end-to-end, tenant-scoped RAG API built with FastAPI, PostgreSQL/pgvector, and optional OpenAI embeddings and generation. The local providers let the full service run without cloud credentials; they are intended for development and demos, not production answer quality.

## Capabilities

- Upload and index TXT, Markdown, CSV, log, and text-based PDF files.
- Split documents into larger parent passages and smaller overlapping child chunks. Embed and retrieve child chunks, then use their parent passage as answer context.
- Store embeddings in PostgreSQL with pgvector and retrieve tenant-scoped matches.
- Ask grounded questions over retrieved context or stream answer tokens over SSE.
- Isolate documents by tenant and optionally bind API keys to individual tenants.
- Delete indexed documents, expose liveness/readiness checks, and publish Prometheus metrics.
- Use the local developer portal to prepare Python, manage Compose services, run tests, and smoke-test the API.
- Run as a non-root container with PostgreSQL health checks and a persistent database volume.

## Run locally

Prerequisites: Docker Desktop with Compose enabled. Copy `.env.example` to `.env`, then run:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The API and interactive OpenAPI docs are at `http://localhost:8000` and `http://localhost:8000/docs`. With the default local providers, no API key is needed; use `X-Tenant-ID: demo` in API calls.
The public pipeline dashboard is at `http://localhost:8000`; its read-only API is under `/api/v1/observability`. Select a tenant, optionally enter its API key, then run the sample pipeline to watch actual upload, embedding, indexing, retrieval, and generation events appear. The event stream is live for requests handled by the current API process.

Ingest a document:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/rag/documents `
  -H "X-Tenant-ID: demo" `
  -F "file=@sample_policy.txt"
```

Ask a question:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/rag/query `
  -H "Content-Type: application/json" -H "X-Tenant-ID: demo" `
  -d '{"query":"What is the critical incident response SLA?","top_k":4}'
```

For real model calls, set `EMBEDDING_PROVIDER=openai`, `LLM_PROVIDER=openai`, and `OPENAI_API_KEY` in `.env`, then recreate the API container. Set `API_KEYS=demo=replace-with-a-long-random-secret` to require a tenant-bound key; include it as `X-API-Key`. In production, configure a unique secret for every tenant, terminate TLS at a trusted ingress, and use managed secret storage.

## Developer portal

The optional local developer portal can prepare the Python environment, start or stop the Compose services, run tests, and exercise the API. It runs as a separate host process, binds only to `127.0.0.1`, and accepts a fixed set of actions; do not deploy it as part of the production API.

Start it from the project root in PowerShell:

```powershell
python tools/dev_portal.py
```

Open `http://127.0.0.1:8765`. **Prepare Python** creates `.venv` and installs `requirements.txt`. **Start services** builds and starts the API and PostgreSQL containers. **Run all tests** invokes `python -m pytest -q` using `.venv` when available. **API smoke test** checks readiness, uploads the included sample policy, verifies a query and citation, then attempts to delete the temporary document. **Validate all** starts services, waits for readiness, runs the tests, and performs the smoke test. **Stop services** runs `docker compose down` without deleting the database volume. Close the terminal or press Ctrl+C to stop the portal.

The portal requires Python on the host; service actions additionally require Docker Desktop with Compose. The first environment preparation and container build may download dependencies/images. If `API_KEYS` is configured for the `demo` tenant in the process environment or `.env`, the smoke test uses that key without displaying it. The portal only binds to loopback and is intended for a trusted local development machine.

## API

- `POST /api/v1/rag/documents`: multipart upload; returns the new document ID and chunk count.
- `POST /api/v1/rag/query`: JSON `{ "query": "...", "top_k": 4 }`; returns answer and cited source chunks.
- `POST /api/v1/rag/stream`: same body; SSE emits `sources`, token `message` events, then `done`.
- `DELETE /api/v1/rag/documents/{document_id}`: delete a tenant's indexed document.
- `GET /api/v1/observability/overview`: public pipeline and stage status, without tenant identifiers or document data.
- `GET /api/v1/observability/events`: recent stage events; use `after` to resume from an event ID.
- `GET /api/v1/observability/stream`: public Server-Sent Events feed with `Last-Event-ID` reconnection support.
- `GET /healthz`: process liveness; `GET /readyz`: database readiness; `GET /metrics`: Prometheus text format.

All RAG routes require `X-Tenant-ID`. When `API_KEYS` is configured, they also require the key bound to that tenant. Keep the API behind authenticated infrastructure in production: this demo does not implement user identity, document ACLs, malware scanning, or audit retention policy.

## Tests

Run the full automated suite from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

The suite covers text and parent-child chunking, ingestion behavior, HTTP API behavior using a fake RAG service, telemetry, and developer-portal action and HTTP protections. Tests use fakes where external services would otherwise be needed, so the automated suite does not require Docker, PostgreSQL, or an OpenAI key. The last verified full run completed with 20 passing tests.

For a service-level check, start Docker Compose and use the developer portal's **API smoke test** or **Validate all** action. The smoke test checks database readiness, uploads the included sample policy, verifies the query answer and citations, then attempts to delete the temporary document. This exercises the live API and PostgreSQL/pgvector path, unlike the isolated automated tests.

## Architecture

Uploads are size-limited and text is extracted before larger parent passages and smaller overlapping child chunks are created. Child text is embedded and indexed, with its parent passage retained as internal metadata and used as answer context. Each indexed row includes `tenant_id`; retrieval and deletion filter by tenant. PostgreSQL stores document metadata and pgvector embeddings. The provider layer selects deterministic local embeddings and extractive answers or OpenAI APIs through environment configuration. Streaming uses SSE with source metadata before generated tokens. The observability ledger records stage status and duration only; it does not retain queries, answer text, tenant IDs, or secrets. It is process-local, bounded, and resets on restart; use a durable telemetry backend for multi-replica or long-term monitoring.

For a production rollout, replace startup `create_all` with versioned Alembic migrations, add OIDC/JWT identity and document-level authorization, evaluate retrieval quality with a curated test set, and set database backups, TLS, resource limits, and secret rotation.
