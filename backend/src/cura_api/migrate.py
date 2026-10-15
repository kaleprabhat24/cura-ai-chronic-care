"""Programmatic Alembic helpers (used by the CLI and tests)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str, settings=None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.attributes["database_url"] = database_url
    cfg.attributes["settings"] = settings  # engine options (SSL, pooler) when provided
    return cfg


def upgrade(database_url: str, revision: str = "head", settings=None, sql: bool = False) -> None:
    command.upgrade(alembic_config(database_url, settings), revision, sql=sql)


def downgrade(database_url: str, revision: str = "base", settings=None) -> None:
    command.downgrade(alembic_config(database_url, settings), revision)
