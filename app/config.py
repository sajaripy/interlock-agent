"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str | None = None
    model: str = "claude-opus-5"
    # Generous on purpose. Opus 5 runs adaptive thinking by default and those
    # tokens count against this cap; a low ceiling truncates a turn mid-thought.
    max_tokens: int = 16_000

    # --- Databases ---------------------------------------------------------
    # One Postgres instance serves two very different consumers:
    #   * the LangGraph checkpointer, which speaks psycopg and owns its own
    #     tables (checkpoints, checkpoint_writes, ...)
    #   * the application tables (tickets, audit_log) via SQLAlchemy + asyncpg
    postgres_dsn: str = "postgresql://interlock:interlock@localhost:5432/interlock"

    # Overrides the SQLAlchemy URL only. Tests point this at sqlite+aiosqlite
    # so the suite runs with no server. Leave unset in production.
    app_db_url: str | None = None

    # --- SMTP (Mailpit in docker-compose) ----------------------------------
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    email_from: str = "agent@interlock.local"

    # --- Policy ------------------------------------------------------------
    policy_path: str = "policy.yaml"

    # --- Observability (all optional) --------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # --- Tool guardrails ---------------------------------------------------
    http_timeout_seconds: float = 10.0
    http_max_bytes: int = 20_000

    @property
    def sqlalchemy_url(self) -> str:
        """Async SQLAlchemy URL for the application tables."""
        if self.app_db_url:
            return self.app_db_url
        return self.postgres_dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

    @property
    def checkpoint_dsn(self) -> str:
        """psycopg DSN for the LangGraph checkpointer."""
        return self.postgres_dsn

    @property
    def policy_file(self) -> Path:
        path = Path(self.policy_path)
        return path if path.is_absolute() else ROOT / path

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests after mutating the environment."""
    get_settings.cache_clear()
