from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Enterprise RAG Service"
    version: str = "1.0.0"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    api_keys: str = ""
    embedding_provider: str = "local"
    llm_provider: str = "local"
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = Field(default=1536, ge=8)
    chunk_size: int = Field(default=900, ge=100, le=5000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    child_chunk_size: int = Field(default=300, ge=32, le=5000)
    child_chunk_overlap: int = Field(default=40, ge=0, le=1000)
    max_upload_bytes: int = Field(default=10_485_760, ge=1024)
    top_k_default: int = Field(default=4, ge=1, le=20)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def api_key_map(self) -> dict[str, str]:
        """Parse tenant=secret pairs into a secret-to-tenant lookup."""
        result: dict[str, str] = {}
        for entry in self.api_keys.split(","):
            tenant, separator, secret = entry.strip().partition("=")
            if separator and tenant and secret:
                result[secret] = tenant
        return result


@lru_cache
def get_settings() -> Settings:
    return Settings()
