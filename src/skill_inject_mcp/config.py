from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    index_lock_timeout_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    lock_poll_s: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    embedding_api_key: str | None = None
    chat_api_key: str | None = None
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    index_summary_chars: int = Field(default=1200, ge=1)
    query_cache_size: int = Field(default=512, ge=0)
    hook_timeout_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    prompt_hook_enabled: bool = False
    http_connect_timeout_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    http_write_timeout_s: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    http_pool_timeout_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    embedding_timeout_s: float = Field(default=EMBEDDING_READ_TIMEOUT_S, gt=0, allow_inf_nan=False)
    embedding_max_concurrency: int = Field(default=2, ge=1)
    embedding_queue_dir: Path | None = None
    embedding_queue_max_wait_s: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    embedding_max_retries: int = Field(default=3, ge=0)
    embedding_retry_statuses: list[int] = Field(default_factory=lambda: [429, 502, 503, 504])
    embedding_retry_base_backoff_s: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    embedding_retry_max_wait_s: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    embedding_retry_jitter: float = Field(default=0.2, ge=0.0, le=1.0, allow_inf_nan=False)
    persistent_embedding_cache: bool = True
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    chat_base_url: str = "https://openrouter.ai/api/v1"
    chat_provider_order: list[str] = Field(default_factory=lambda: ["Cerebras", "Groq"])
    use_fake_embedder: bool = False
    retrieval_mode: Literal["hybrid", "lexical"] = "hybrid"
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

    verification_mode: Literal["semantic", "lexical"] = "semantic"
    verification_model: str = "openai/gpt-oss-120b"
    verification_top_k: int = Field(default=5, ge=1, le=20)
    verification_timeout_s: float = Field(default=VERIFICATION_READ_TIMEOUT_S, gt=0, allow_inf_nan=False)
    verification_max_tokens: int = Field(default=8192, ge=1)
    verification_max_retries: int = Field(default=1, ge=0, le=1)
    verification_max_source_chars: int = Field(default=16000, ge=1)
    verification_cache_size: int = Field(default=256, ge=0)

    @model_validator(mode="after")
    def validate_retrieval_mode(self):
        if self.retrieval_mode == "lexical" and self.verification_mode != "lexical":
            raise ValueError("lexical retrieval requires lexical verification")
        return self


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

    def resolve_embedding_api_key(self) -> str | None:
        if self.embedding_api_key and self.embedding_api_key.strip():
            return self.embedding_api_key.strip()
        return self.resolve_api_key()

    def resolve_chat_api_key(self) -> str | None:
        if self.chat_api_key and self.chat_api_key.strip():
            return self.chat_api_key.strip()
        return self.resolve_api_key()

    def multi_query_enabled(self) -> bool:
        """True when multi-query is configured on and a chat API key is available."""
        return self.retrieval_mode == "hybrid" and bool(self.multi_query) and bool(self.resolve_chat_api_key())


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)


def load_hook_settings(repo_root: Path | None = None) -> Settings:
    """Load Settings anchored to repo_root regardless of the caller's working directory."""
    import os
    from dotenv import dotenv_values

    anchor = (repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    env_file = anchor / ".env"
    raw_env = dotenv_values(env_file, interpolate=False) if env_file.is_file() else {}

    def _has_cfg(key: str) -> bool:
        prefixed = f"SKILL_INJECT_{key.upper()}"
        return (
            prefixed in os.environ
            or prefixed in raw_env
            or key.upper() in os.environ
            or key.upper() in raw_env
        )

    overrides: dict[str, object] = {"_env_file": env_file if env_file.is_file() else None}
    if not _has_cfg("persistent_embedding_cache"):
        overrides["persistent_embedding_cache"] = True
    if not _has_cfg("hook_timeout_s"):
        overrides["hook_timeout_s"] = 4.0

    settings = Settings(**overrides)

    def _resolve_rel(p: Path | None) -> Path | None:
        if p is None:
            return None
        expanded = p.expanduser()
        return expanded if expanded.is_absolute() else (anchor / expanded).resolve()

    settings.skills_dir = _resolve_rel(settings.skills_dir) or settings.skills_dir
    settings.skill_manifest = _resolve_rel(settings.skill_manifest)
    settings.index_dir = _resolve_rel(settings.index_dir) or settings.index_dir
    settings.embedding_queue_dir = _resolve_rel(settings.embedding_queue_dir)
    settings.sqlite_vector_path = _resolve_rel(settings.sqlite_vector_path)
    settings.openrouter_api_key_file = _resolve_rel(settings.openrouter_api_key_file)
    return settings
