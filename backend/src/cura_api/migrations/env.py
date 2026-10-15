"""Alembic environment. The database URL comes from `config.attributes["database_url"]` (set by
`cura_api.migrate`) or, for the plain `alembic` CLI, from the application settings (DATABASE_URL)."""

from __future__ import annotations

from alembic import context

from cura_api import models
from cura_api.db import UTCDateTime, engine_from_settings, make_engine
from cura_api.settings import get_settings

config = context.config
url = config.attributes.get("database_url") or get_settings().database_url
target_metadata = models.Base.metadata


def render_item(type_, obj, autogen_context):
    # Store UTCDateTime as a plain timezone-aware DateTime so migrations don't import app code.
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True, render_item=render_item)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    settings = config.attributes.get("settings")
    engine = engine_from_settings(settings) if settings is not None else make_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=True, render_item=render_item, compare_type=False)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
