"""`cura-api` command line: database migrations, cohort loading, batch scoring, server, OpenAPI export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cura_api.settings import get_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cura-api", description="Cura AI backend tools")
    sub = parser.add_subparsers(dest="command", required=True)
    db = sub.add_parser("db", help="database migrations")
    db.add_argument("action", choices=["upgrade", "downgrade"])
    db.add_argument("--revision", default=None)
    seed = sub.add_parser("seed", help="load the generated synthetic cohort (idempotent)")
    seed.add_argument("--data", default=None, help="dataset directory (default DATASET_DIR)")
    score = sub.add_parser("score", help="batch-score eligible episodes with the saved model bundle")
    score.add_argument("--scope", choices=["eligible", "open"], default="eligible")
    score.add_argument("--bundle", default=None, help="model bundle directory (default MODEL_BUNDLE_DIR)")
    score.add_argument("--data", default=None)
    score.add_argument("--allow-fingerprint-mismatch", action="store_true")
    serve = sub.add_parser("serve", help="run the API with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    oa = sub.add_parser("openapi", help="write the OpenAPI document")
    oa.add_argument("--out", default="backend/openapi.json")
    users = sub.add_parser("users", help="provision application users and roles (Supabase mode)")
    users.add_argument("action", choices=["add", "list", "set-role", "deactivate", "activate"])
    users.add_argument("email", nargs="?")
    users.add_argument("--role", choices=["coordinator", "analyst", "admin"])
    users.add_argument("--name", default=None)
    fx = sub.add_parser("fhir-export", help="write a patient's FHIR R4-oriented Bundle and validate it")
    fx.add_argument("patient_id")
    fx.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    started = time.perf_counter()
    if args.command == "db":
        from cura_api import migrate

        if args.action == "upgrade":
            migrate.upgrade(settings.database_url, args.revision or "head", settings=settings)
        else:
            migrate.downgrade(settings.database_url, args.revision or "base", settings=settings)
        print(f"database {args.action} done: {_redact(settings.database_url)}")
    elif args.command in ("seed", "score"):
        from cura_api.db import engine_from_settings, make_session_factory

        engine = engine_from_settings(settings)
        factory = make_session_factory(engine)
        if args.command == "seed":
            from cura_api.services.seed import seed_database

            result = seed_database(factory, Path(args.data or settings.dataset_dir))
        else:
            from cura_api.services.scoring import run_scoring

            result = run_scoring(factory, Path(args.bundle or settings.model_bundle_dir),
                                 Path(args.data or settings.dataset_dir), scope=args.scope,
                                 allow_fingerprint_mismatch=args.allow_fingerprint_mismatch)
        engine.dispose()
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "users":
        from cura_api.db import engine_from_settings, make_session_factory
        from cura_api.services import users as u

        engine = engine_from_settings(settings)
        factory = make_session_factory(engine)
        try:
            if args.action == "list":
                result = u.list_users(factory)
            elif not args.email:
                parser.error("email is required")
            elif args.action == "add":
                if not args.role:
                    parser.error("--role is required")
                result = u.add_user(factory, args.email, args.role, args.name)
            elif args.action == "set-role":
                if not args.role:
                    parser.error("--role is required")
                result = u.update_user(factory, args.email, role=args.role)
            else:
                result = u.update_user(factory, args.email, active=args.action == "activate")
        except u.UserError as exc:
            parser.exit(2, f"error: {exc}" + chr(10))
        engine.dispose()
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "fhir-export":
        from cura_api.api.common import latest_load
        from cura_api.db import engine_from_settings, make_session_factory
        from cura_api.services.fhir import build_bundle, r4b_parse_errors, structural_problems

        engine = engine_from_settings(settings)
        with make_session_factory(engine)() as session:
            bundle = build_bundle(session, args.patient_id, latest_load(session).data_cutoff)
        engine.dispose()
        out = Path(args.out or f"cura-ai-{args.patient_id}-fhir-bundle.json")
        out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        problems, parse = structural_problems(bundle), r4b_parse_errors(bundle)
        print(f"{len(bundle['entry'])} resources -> {out}")
        print(f"structural checks: {problems or 'ok'}")
        print(f"fhir.resources R4B model parsing: {parse or 'ok'}")
        print("note: not validated with the official HL7 FHIR validator (see docs/FHIR_EXPORT.md)")
    elif args.command == "serve":
        import uvicorn

        settings.check_runtime()
        uvicorn.run("cura_api.main:create_app", factory=True, host=args.host, port=args.port,
                    reload=args.reload)
    else:
        from cura_api.main import create_app

        doc = create_app(settings).openapi()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"OpenAPI {doc['openapi']} with {len(doc['paths'])} paths -> {args.out}")
    print(f"{args.command} finished in {time.perf_counter() - started:.1f}s")


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


if __name__ == "__main__":
    main()
