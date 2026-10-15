"""Database engine, sessions and portable column types (SQLite by default, PostgreSQL via DATABASE_URL)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps on every backend. SQLite has no timezone support, so values are
    stored as naive UTC there and re-attached to UTC on read. Naive input is rejected."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed; use timezone-aware UTC")
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def engine_options(url: str, echo: bool = False, *, pool_size: int = 5, max_overflow: int = 5,
                   pool_recycle: int = 1800, sslmode: str | None = None,
                   disable_prepared_statements: bool = False) -> dict:
    """create_engine keyword arguments. PostgreSQL gets a pre-pinged, recycled connection pool, optional
    `sslmode` (Supabase: "require") and, for PgBouncer/transaction poolers, no prepared statements."""
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        return kwargs
    connect_args: dict = {}
    if sslmode:
        connect_args["sslmode"] = sslmode
    if disable_prepared_statements:
        connect_args["prepare_threshold"] = None  # psycopg 3: never prepare (PgBouncer transaction mode)
    kwargs.update(pool_pre_ping=True, pool_size=pool_size, max_overflow=max_overflow,
                  pool_recycle=pool_recycle, connect_args=connect_args)
    return kwargs


def make_engine(url: str, echo: bool = False, **options) -> Engine:
    kwargs = engine_options(url, echo, **options)
    if url.startswith("sqlite"):
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    return engine


def engine_from_settings(settings) -> Engine:
    return make_engine(settings.database_url, settings.sql_echo, pool_size=settings.database_pool_size,
                       max_overflow=settings.database_max_overflow,
                       pool_recycle=settings.database_pool_recycle_seconds,
                       sslmode=settings.database_sslmode,
                       disable_prepared_statements=settings.database_disable_prepared_statements)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def utcnow() -> datetime:
    return datetime.now(UTC)
