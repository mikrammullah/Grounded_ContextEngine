from collections.abc import AsyncIterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import session_factory
from app.services.rag_service import RAGService


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def get_tenant_id(
    x_tenant_id: str = Header(min_length=1, max_length=120),
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    configured_keys = settings.api_key_map
    if configured_keys:
        tenant_for_key = configured_keys.get(x_api_key or "")
        if tenant_for_key is None or tenant_for_key != x_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or tenant",
            )
    elif settings.environment == "production":
        raise HTTPException(status_code=500, detail="API key authentication is not configured")
    return x_tenant_id


async def get_rag_service(
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> RAGService:
    return RAGService(session=session, settings=settings)
