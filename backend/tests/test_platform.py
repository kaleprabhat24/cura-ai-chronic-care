"""Configuration safety, PostgreSQL compatibility and the new list/document endpoints."""

from __future__ import annotations

import pytest
from conftest import ADMIN, ANALYST, COORD, make_settings

from cura_api import migrate
from cura_api.db import engine_options
from cura_api.main import create_app
from cura_api.settings import Settings, normalise_database_url

PG = "postgresql+psycopg://cura:secret@db.example.supabase.co:5432/postgres"


def test_start_up_refusals(env):
    base = (env.root, env.database_url, env.data_dir, env.bundle_dir)
    with pytest.raises(RuntimeError, match="refuses to run"):
        create_app(make_settings(*base, environment="production"))
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        create_app(make_settings(*base, auth_mode="supabase"))
    with pytest.raises(RuntimeError, match="SUPABASE_URL"):
        create_app(make_settings(*base, auth_mode="supabase", supabase_url="http://insecure.example"))
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        make_settings(*base, environment="production", auth_mode="supabase",
                      supabase_url="https://x.supabase.co").check_runtime()
    ok = Settings(_env_file=None, environment="production", auth_mode="supabase",
                  supabase_url="https://x.supabase.co/", database_url="postgres://u:p@h:5432/db")
    ok.check_runtime()  # valid production configuration (no connection is made here)
    assert ok.database_url.startswith("postgresql+psycopg://")
    assert ok.supabase_issuer == "https://x.supabase.co/auth/v1"
    assert ok.resolved_jwks_url == "https://x.supabase.co/auth/v1/.well-known/jwks.json"


def test_database_url_normalisation_and_postgres_engine_options():
    assert normalise_database_url("postgres://a:b@h/db") == "postgresql+psycopg://a:b@h/db"
    assert normalise_database_url("postgresql://a:b@h/db") == "postgresql+psycopg://a:b@h/db"
    assert normalise_database_url("sqlite:///x.db") == "sqlite:///x.db"
    opts = engine_options(PG, pool_size=3, max_overflow=2, sslmode="require",
                          disable_prepared_statements=True)
    assert opts["pool_pre_ping"] and opts["pool_size"] == 3 and opts["max_overflow"] == 2
    assert opts["connect_args"] == {"sslmode": "require", "prepare_threshold": None}
    assert engine_options("sqlite:///x.db")["connect_args"] == {"check_same_thread": False}


def test_migrations_render_for_postgresql(capsys):
    """Offline (SQL script) rendering of every migration for the PostgreSQL dialect."""
    migrate.upgrade(PG, "head", sql=True)
    sql = capsys.readouterr().out
    for table in ("patients", "followup_episodes", "risk_scores", "outreach_attempts", "audit_events",
                  "app_users", "scoring_runs"):
        assert f"CREATE TABLE {table}" in sql, table
    assert "TIMESTAMP WITH TIME ZONE" in sql and "JSON" in sql
    assert "UPDATE alembic_version SET version_num='0003'" in sql
    assert 'ALTER TABLE public."risk_scores" ENABLE ROW LEVEL SECURITY' in sql


def test_outreach_list_filters_and_permissions(client):
    item = client.get("/api/v1/worklist?page_size=1&queue=predictive", headers=COORD).json()["items"][0]
    client.post(f"/api/v1/episodes/{item['episode_id']}/outreach-attempts", headers=COORD,
                json={"channel": "phone", "outcome": "left_message", "next_action": "retry",
                      "next_action_due": "2030-02-01"})
    app = client.get("/api/v1/outreach-attempts?origin=app&page_size=100", headers=COORD).json()
    assert app["total"] >= 1 and all(o["origin"] == "app" for o in app["items"])
    assert all(o["patient"]["display_name"] for o in app["items"])
    pending = client.get("/api/v1/outreach-attempts?pending_next_action=true&sort=next_action_due"
                         "&page_size=50", headers=COORD).json()
    assert all(o["next_action"] != "none" for o in pending["items"])
    dues = [o["next_action_due"] for o in pending["items"] if o["next_action_due"]]
    assert dues == sorted(dues)
    only = client.get("/api/v1/outreach-attempts?outcome=left_message&page_size=20", headers=COORD).json()
    assert all(o["outcome"] == "left_message" for o in only["items"])
    assert client.get("/api/v1/outreach-attempts", headers=ANALYST).status_code == 403
    bad = client.get("/api/v1/outreach-attempts?outcome=letter_sent", headers=COORD)
    assert bad.status_code == 422


def test_model_documents_info_and_admin_users(client):
    docs = client.get("/api/v1/models/current/documents", headers=ANALYST).json()
    assert docs["model_version"] == "test-api" and "Model card" in docs["model_card_markdown"]
    info = client.get("/api/v1/models/current", headers=COORD).json()
    assert len(info["calibration_table"]) == 10 and info["test_tiers"] and info["feature_importance"]
    assert client.get("/api/v1/admin/users", headers=ADMIN).status_code == 200
    assert client.get("/api/v1/admin/users", headers=COORD).status_code == 403
    out = client.get("/api/v1/outreach/outcomes", headers=ANALYST).json()
    months = out["attempts_by_month"]
    assert months and {"month", "origin", "group", "attempts"} <= set(months[0])


def test_analyst_payloads_contain_no_patient_names(client):
    for path in ("/dashboard/summary", "/outreach/outcomes", "/models/current", "/data-quality",
                 "/scoring-runs"):
        text = client.get(f"/api/v1{path}", headers=ANALYST).text
        assert "given_name" not in text and "family_name" not in text and "display_name" not in text


def test_unfilled_env_example_falls_back_to_safe_local_defaults():
    from cura_api.settings import PROJECT_ROOT

    s = Settings(_env_file=PROJECT_ROOT / ".env.example")
    s.check_runtime()
    assert (s.environment, s.auth_mode) == ("development", "dev")
    assert s.is_sqlite and s.cors_origins == ["http://localhost:3000"]
    text = (PROJECT_ROOT / ".env.example").read_text("utf-8")
    assert all(line.endswith("=") for line in text.splitlines() if line and not line.startswith("#"))
