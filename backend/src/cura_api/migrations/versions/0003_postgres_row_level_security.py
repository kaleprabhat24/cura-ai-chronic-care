"""enable row level security on PostgreSQL (Supabase Data API lockdown)

Supabase exposes tables in the `public` schema through its Data API to anyone holding the public
anon key. Enabling RLS with *no policies* denies all rows to the `anon` and `authenticated` API
roles. The backend connects as the table owner, which RLS does not restrict, so the application keeps
working. SQLite has no RLS, so this migration does nothing there.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15 20:00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("patients", "conditions", "appointments", "encounters", "observations", "medication_statements",
          "followup_episodes", "episode_outcomes", "outreach_attempts", "dataset_loads", "scoring_runs",
          "risk_scores", "audit_events", "app_users", "alembic_version")


def upgrade() -> None:
    if op.get_context().dialect.name != "postgresql":
        return
    for table in TABLES:
        op.execute(f'ALTER TABLE public."{table}" ENABLE ROW LEVEL SECURITY')


def downgrade() -> None:
    if op.get_context().dialect.name != "postgresql":
        return
    for table in TABLES:
        op.execute(f'ALTER TABLE public."{table}" DISABLE ROW LEVEL SECURITY')
