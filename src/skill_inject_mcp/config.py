from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    index_dir: Path = Field(
        default_factory=lambda: Path(".skill_inject_index"),
        description="Directory for SQLite indexes",
    )
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dim: int = 1024
    embedding_base_url: str = "https://openrouter.ai/api/v1"
    use_fake_embedder: bool = False
    rrf_k: int = 60
    retrieve_top_k: int = 20
    rerank: Literal["off", "qwen3-0.6b"] = "off"

    def resolve_api_key(self) -> str | None:
        # Also accept bare OPENROUTER_API_KEY via env without prefix
        import os

        return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
