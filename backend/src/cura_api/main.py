"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from cura_api import __version__, errors
from cura_api.api import core, insights, onboarding, patients, worklist
from cura_api.auth import build_verifier
from cura_api.db import engine_from_settings, make_session_factory, utcnow
from cura_api.settings import Settings, get_settings

API_PREFIX = "/api/v1"
DESCRIPTION = """
Backend for **Cura AI**, a chronic-disease follow-up prioritisation demo. **All data is synthetic.**

* Risk scores are *simulated predictions* produced by batch scoring runs with the saved model bundle.
  They are never recomputed when a page is opened.
* Every payload labels its provenance: `simulated_prediction`, `recorded_synthetic_clinical`,
  `recorded_synthetic_outcome` or `recorded_outreach_outcome`.
* **Authentication:** `AUTH_MODE=supabase` verifies Supabase Auth access tokens
  (`Authorization: Bearer ...`) and maps them to provisioned `app_users` roles (`coordinator`,
  `analyst`, `admin`). `AUTH_MODE=dev` (local only, refused in production) accepts `X-Dev-User` /
  `X-Dev-Role` headers.
* Errors have the shape `{"error": {"code", "message", "details"}}`.
"""


def create_app(settings: Settings | None = None,
               session_factory: sessionmaker[Session] | None = None,
               token_verifier=None, clock=None) -> FastAPI:
    settings = settings or get_settings()
    settings.check_runtime()
    engine = None
    if session_factory is None:
        engine = engine_from_settings(settings)
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
    app.state.clock = clock or utcnow  # real UTC time; tests may inject a fixed clock
    app.state.token_verifier = token_verifier or (build_verifier(settings)
                                                  if settings.auth_mode == "supabase" else None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type", "X-Dev-User", "X-Dev-Role"])
    errors.install(app)
    for router in (core.router, worklist.router, onboarding.router, patients.router, insights.router):
        app.include_router(router, prefix=API_PREFIX)
    return app
