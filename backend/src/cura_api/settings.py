"""Application settings (environment variables or a `.env` file at the project root).

Safety rules enforced at start-up (`Settings.check_runtime`):
  * development authentication (`AUTH_MODE=dev`, identity headers) refuses to run when
    `ENVIRONMENT=production`;
  * `AUTH_MODE=supabase` requires `SUPABASE_URL` (access tokens are verified against that project's
    published signing keys, or the legacy JWT secret if configured);
  * `ENVIRONMENT=production` requires PostgreSQL (a SQLite file on an ephemeral host would lose data).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = PROJECT_ROOT / "backend" / "var" / "cura_dev.db"


def normalise_database_url(url: str) -> str:
    """Accept the URI formats Supabase/Render print (`postgres://`, `postgresql://`) and use the
    psycopg 3 driver for them."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Settings(BaseSettings):
    # Empty variables (e.g. a copied, unfilled .env.example) fall back to the defaults.
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8",
                                      extra="ignore", env_ignore_empty=True)

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = Field(default=f"sqlite:///{DEFAULT_DB.as_posix()}")
    # PostgreSQL connection options (ignored for SQLite)
    database_sslmode: Literal["disable", "allow", "prefer", "require", "verify-ca",
                              "verify-full"] | None = None
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=5, ge=0, le=50)
    database_pool_recycle_seconds: int = Field(default=1800, ge=30)
    # Supabase transaction pooler (port 6543 / PgBouncer): server-side prepared statements must be off
    database_disable_prepared_statements: bool = False

    auth_mode: Literal["dev", "supabase"] = "dev"
    supabase_url: str | None = None                # https://<project-ref>.supabase.co
    supabase_jwt_audience: str = "authenticated"
    supabase_jwks_url: str | None = None           # default: <SUPABASE_URL>/auth/v1/.well-known/jwks.json
    supabase_jwt_secret: SecretStr | None = None   # only for projects still on the legacy HS256 secret

    dataset_dir: Path = PROJECT_ROOT / "ml" / "data" / "v1"
    cohort_config: Path = PROJECT_ROOT / "ml" / "configs" / "cohort_v1.yaml"
    model_bundle_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "models" / "cura-fu-1.0.0"
    model_artifacts_root: Path = PROJECT_ROOT / "ml" / "artifacts" / "models"
    model_reports_dir: Path = PROJECT_ROOT / "ml" / "reports"
    cors_origins: list[str] = ["http://localhost:3000"]
    sql_echo: bool = False

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Relative paths (and relative SQLite files) are resolved against the project root, so
        commands behave the same from any working directory."""
        for name in ("dataset_dir", "cohort_config", "model_bundle_dir", "model_artifacts_root",
                     "model_reports_dir"):
            p = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, PROJECT_ROOT / p)
        self.database_url = normalise_database_url(self.database_url)
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            path = self.database_url[len(prefix):]
            if path and path != ":memory:" and not Path(path).is_absolute():
                self.database_url = prefix + (PROJECT_ROOT / path).as_posix()
        if self.supabase_url:
            self.supabase_url = self.supabase_url.rstrip("/")
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def supabase_issuer(self) -> str | None:
        return f"{self.supabase_url}/auth/v1" if self.supabase_url else None

    @property
    def resolved_jwks_url(self) -> str | None:
        if self.supabase_jwks_url:
            return self.supabase_jwks_url
        return f"{self.supabase_issuer}/.well-known/jwks.json" if self.supabase_issuer else None

    def check_runtime(self) -> None:
        if self.auth_mode == "dev" and self.environment == "production":
            raise RuntimeError("AUTH_MODE=dev is a local development mode and refuses to run with "
                               "ENVIRONMENT=production.")
        if self.auth_mode == "supabase" and not (self.supabase_url or "").startswith("https://"):
            raise RuntimeError("AUTH_MODE=supabase requires SUPABASE_URL=https://<project-ref>.supabase.co")
        if self.environment == "production" and self.is_sqlite:
            raise RuntimeError("ENVIRONMENT=production requires a PostgreSQL DATABASE_URL.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
