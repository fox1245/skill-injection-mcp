from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from skill_inject_mcp.timeouts import (
    EMBEDDING_READ_TIMEOUT_S, MULTI_QUERY_READ_TIMEOUT_S, VERIFICATION_READ_TIMEOUT_S,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SKILL_INJECT_",
        env_file=".env",
        extra="ignore",
    )

    skills_dir: Path = Field(
        default_factory=lambda: Path("fixtures/skills"),
        description="Root directory containing skill folders with SKILL.md",
    )
    skill_manifest: Path | None = None

    index_dir: Path = Field(
        default_factory=lambda: Path(".skill_inject_index"),
        description="Directory for SQLite indexes",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    query_cache_size: int = Field(default=512, ge=0)
    hook_timeout_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    embedding_timeout_s: float = Field(default=EMBEDDING_READ_TIMEOUT_S, gt=0, allow_inf_nan=False)
    persistent_embedding_cache: bool = False
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    use_fake_embedder: bool = False
    dense_backend: Literal["sqlite-vector", "numpy"] = "sqlite-vector"
    sqlite_vector_path: Path | None = None
    openrouter_api_key_file: Path | None = Field(
        default=None, description="Authorized dotenv file containing OPENROUTER_API_KEY; never log its contents",
    )
    rrf_k: int = Field(default=60, ge=1)
    retrieve_top_k: int = Field(default=20, ge=1)
    # Returned evidence width; acceptance uses the complete retrieved candidate pool.
    resolve_top_k: int = Field(default=5, ge=1)
    rerank: Literal["off", "qwen3-0.6b"] = "off"
    # Multi-query expansion (OpenRouter chat). Effective only when API key present.
    multi_query: bool = True
    multi_query_model: str = "openai/gpt-oss-120b"
    multi_query_timeout_s: float = Field(default=MULTI_QUERY_READ_TIMEOUT_S, gt=0, allow_inf_nan=False)

    verification_mode: Literal["semantic", "lexical"] = "lexical"
    verification_model: str = "openai/gpt-oss-120b"
    verification_top_k: int = Field(default=5, ge=1, le=20)
    verification_timeout_s: float = Field(default=VERIFICATION_READ_TIMEOUT_S, gt=0, allow_inf_nan=False)
    verification_max_tokens: int = Field(default=8192, ge=1)
    verification_max_retries: int = Field(default=1, ge=0, le=1)
    verification_max_source_chars: int = Field(default=16000, ge=1)
    verification_cache_size: int = Field(default=256, ge=0)

    def resolve_api_key(self) -> str | None:
        # Also accept bare OPENROUTER_API_KEY via env without prefix
        import os

        if self.openrouter_api_key_file is not None:
            from dotenv import dotenv_values
            source = self.openrouter_api_key_file.expanduser().resolve()
            if not source.is_file():
                raise ValueError("Configured OpenRouter key file does not exist")
            key = dotenv_values(source, interpolate=False).get("OPENROUTER_API_KEY")
            if not key or not key.strip():
                raise ValueError("Configured OpenRouter key file has no OPENROUTER_API_KEY")
            return key.strip()
        return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    def multi_query_enabled(self) -> bool:
        """True when multi-query is configured on and an API key is available."""
        return bool(self.multi_query) and bool(self.resolve_api_key())


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
