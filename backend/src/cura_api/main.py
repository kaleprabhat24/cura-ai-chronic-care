"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from cura_api import __version__, errors
from cura_api.api import core, insights, patients, worklist
from cura_api.db import make_engine, make_session_factory
from cura_api.settings import Settings, get_settings

API_PREFIX = "/api/v1"
DESCRIPTION = """
Backend for **Cura AI**, a chronic-disease follow-up prioritisation demo. **All data is synthetic.**

* Risk scores are *simulated predictions* produced by batch scoring runs with the saved model bundle.
  They are never recomputed when a page is opened.
* Every payload labels its provenance: `simulated_prediction`, `recorded_synthetic_clinical`,
  `recorded_synthetic_outcome` or `recorded_outreach_outcome`.
* **Authentication:** only a local development mode exists (`X-Dev-User` / `X-Dev-Role` headers;
  roles `coordinator`, `analyst`, `admin`). It refuses to run in production. Production
  authentication is not implemented yet.
* Errors have the shape `{"error": {"code", "message", "details"}}`.
"""


def create_app(settings: Settings | None = None,
               session_factory: sessionmaker[Session] | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.check_runtime()
    engine = None
    if session_factory is None:
        engine = make_engine(settings.database_url, settings.sql_echo)
        session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if engine is not None:
            engine.dispose()

    app = FastAPI(title="Cura AI API", version=__version__, description=DESCRIPTION, lifespan=lifespan,
                  openapi_tags=[{"name": n} for n in ("system", "dashboard", "worklist", "patients",
                                                      "episodes", "risk scores", "outreach", "model",
                                                      "admin")])
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    errors.install(app)
    for router in (core.router, worklist.router, patients.router, insights.router):
        app.include_router(router, prefix=API_PREFIX)
    return app
