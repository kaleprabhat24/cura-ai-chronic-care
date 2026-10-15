"""Application settings (environment variables or a `.env` file at the project root).

Safety rules enforced at start-up (`Settings.check_runtime`):
  * development authentication (`AUTH_MODE=dev`) refuses to run when `ENVIRONMENT=production`;
  * `AUTH_MODE=supabase` is reserved for the final milestone and refuses to start. Production
    authentication is **not implemented yet**, and the service does not pretend otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = PROJECT_ROOT / "backend" / "var" / "cura_dev.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8",
                                      extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = Field(default=f"sqlite:///{DEFAULT_DB.as_posix()}")
    auth_mode: Literal["dev", "supabase"] = "dev"
    dataset_dir: Path = PROJECT_ROOT / "ml" / "data" / "v1"
    cohort_config: Path = PROJECT_ROOT / "ml" / "configs" / "cohort_v1.yaml"
    model_bundle_dir: Path = PROJECT_ROOT / "ml" / "artifacts" / "models" / "cura-fu-1.0.0"
    model_artifacts_root: Path = PROJECT_ROOT / "ml" / "artifacts" / "models"
    cors_origins: list[str] = ["http://localhost:3000"]
    sql_echo: bool = False

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Relative paths (and relative SQLite files) are resolved against the project root, so
        commands behave the same from any working directory."""
        for name in ("dataset_dir", "cohort_config", "model_bundle_dir", "model_artifacts_root"):
            p = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, PROJECT_ROOT / p)
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            path = self.database_url[len(prefix):]
            if path and path != ":memory:" and not Path(path).is_absolute():
                self.database_url = prefix + (PROJECT_ROOT / path).as_posix()
        return self

    def check_runtime(self) -> None:
        if self.auth_mode == "dev" and self.environment == "production":
            raise RuntimeError("AUTH_MODE=dev is a local development mode and refuses to run with "
                               "ENVIRONMENT=production.")
        if self.auth_mode == "supabase":
            raise RuntimeError("AUTH_MODE=supabase is not implemented yet (planned for the final "
                               "milestone). Use AUTH_MODE=dev for local development.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
