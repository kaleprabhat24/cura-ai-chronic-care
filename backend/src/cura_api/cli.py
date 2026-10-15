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
    args = parser.parse_args(argv)

    settings = get_settings()
    started = time.perf_counter()
    if args.command == "db":
        from cura_api import migrate

        if args.action == "upgrade":
            migrate.upgrade(settings.database_url, args.revision or "head")
        else:
            migrate.downgrade(settings.database_url, args.revision or "base")
        print(f"database {args.action} done: {_redact(settings.database_url)}")
    elif args.command in ("seed", "score"):
        from cura_api.db import make_engine, make_session_factory

        engine = make_engine(settings.database_url, settings.sql_echo)
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
