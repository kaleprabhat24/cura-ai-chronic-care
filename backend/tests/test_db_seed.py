"""Migrations, persistence and idempotent cohort loading."""

from __future__ import annotations

import shutil

import pandas as pd
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import func, inspect, select

from cura_api import migrate
from cura_api import models as m
from cura_api.db import make_engine, make_session_factory, utcnow
from cura_api.services.seed import seed_database
from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import read_table


def test_migrations_create_schema_matching_models(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    migrate.upgrade(url)
    engine = make_engine(url)
    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())
        assert set(m.Base.metadata.tables) <= tables
        diffs = compare_metadata(MigrationContext.configure(conn), m.Base.metadata)
    assert diffs == [], diffs
    engine.dispose()
    migrate.downgrade(url)
    engine = make_engine(url)
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) <= {"alembic_version"}
    engine.dispose()


def test_seed_loaded_every_source_row(env):
    engine, factory = env.factory()
    with factory() as s:
        for model, group, table in (
            (m.Patient, "records", "patients"), (m.Condition, "records", "conditions"),
            (m.Appointment, "records", "appointments"), (m.Encounter, "records", "encounters"),
            (m.Observation, "records", "observations"),
            (m.MedicationStatement, "records", "medication_statements"),
            (m.FollowupEpisode, "cohort", "followup_episodes"),
            (m.EpisodeOutcome, "outcomes", "episode_labels"),
        ):
            n = s.execute(select(func.count()).select_from(model)).scalar_one()
            assert n == len(read_table(env.data_dir, group, table)), table
        n_hist = s.execute(select(func.count()).select_from(m.OutreachAttempt).where(
            m.OutreachAttempt.origin == "historical_simulation")).scalar_one()
        assert n_hist == len(read_table(env.data_dir, "records", "outreach_attempts"))
    first = env.seed_result["tables"]
    assert all(v["inserted"] == v["source_rows"] and v["updated"] == 0 for v in first.values())
    engine.dispose()


def test_seed_values_round_trip(env):
    engine, factory = env.factory()
    src = read_table(env.data_dir, "records", "observations").iloc[123]
    with factory() as s:
        o = s.get(m.Observation, src["observation_id"])
        assert o.value_num == src["value_num"] and o.unit == src["unit"]
        assert o.effective_at == as_utc(pd.Series([src["effective_at"]])).iloc[0].to_pydatetime()
        assert o.effective_at.tzinfo is not None
        ep = read_table(env.data_dir, "cohort", "followup_episodes").iloc[50]
        row = s.get(m.FollowupEpisode, ep["episode_id"])
        assert row.due_date == ep["due_date"] and row.eligible == bool(ep["eligible"])
    engine.dispose()


def test_reseed_is_idempotent_and_reports_counts(env, tmp_path):
    db = tmp_path / "copy.db"
    shutil.copy(env.root / "cura_test.db", db)
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    factory = make_session_factory(engine)
    again = seed_database(factory, env.data_dir, actor="system:test")
    for name, c in again["tables"].items():
        assert c["inserted"] == 0 and c["updated"] == 0, name
        assert c["skipped"] == c["source_rows"], name
    # change one source value -> exactly one update; app-entered outreach is never touched
    with factory() as s, s.begin():
        ep = s.execute(select(m.FollowupEpisode)
                       .where(m.FollowupEpisode.eligible.is_(True))).scalars().first()
        s.add(m.OutreachAttempt(id="oa_app_keep", patient_id=ep.patient_id, episode_id=ep.id,
                                programme=ep.programme, origin="app", purpose="coordinator_outreach",
                                channel="phone", outcome="no_answer", next_action="none",
                                attempted_at=utcnow(), recorded_at=utcnow(), recorded_by="x"))
    data = tmp_path / "data"
    shutil.copytree(env.data_dir, data)
    pts = read_table(data, "records", "patients")
    pts.loc[0, "preferred_language"] = "xx"
    pts.to_parquet(data / "records" / "patients.parquet", index=False)
    changed = seed_database(factory, data, actor="system:test")
    assert changed["tables"]["patients"]["updated"] == 1
    assert changed["tables"]["patients"]["skipped"] == len(pts) - 1
    assert changed["tables"]["outreach_attempts"]["not_in_source"] == 0
    with factory() as s:
        assert s.get(m.Patient, pts.loc[0, "patient_id"]).preferred_language == "xx"
        assert s.get(m.OutreachAttempt, "oa_app_keep") is not None
        assert s.execute(select(func.count()).select_from(m.DatasetLoad)).scalar_one() == 3
    engine.dispose()


def test_historical_outreach_mapping_and_episode_links(env):
    engine, factory = env.factory()
    with factory() as s:
        letters = s.execute(select(m.OutreachAttempt)
                            .where(m.OutreachAttempt.channel == "letter")).scalars().all()
        assert letters and all(o.outcome == "other" and "letter" in o.note.lower() for o in letters)
        outcomes = set(s.execute(select(m.OutreachAttempt.outcome).distinct()).scalars())
        assert outcomes <= {"no_answer", "left_message", "reached_booked", "reached_declined",
                            "reached_already_done", "wrong_number", "other"}
        rows = s.execute(select(m.OutreachAttempt, m.FollowupEpisode)
                         .join(m.FollowupEpisode, m.FollowupEpisode.id == m.OutreachAttempt.episode_id)
                         .limit(500)).all()
        assert rows
        for o, ep in rows:
            assert ep.patient_id == o.patient_id and ep.programme == o.programme
            assert ep.index_event_at <= o.attempted_at
    engine.dispose()
