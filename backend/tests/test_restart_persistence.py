"""Integration: a recorded outreach outcome survives a fresh database session and an API restart."""

from __future__ import annotations

import shutil

from conftest import COORD, make_settings
from fastapi.testclient import TestClient
from sqlalchemy import select

from cura_api import models as m
from cura_api.db import make_engine, make_session_factory
from cura_api.main import create_app


def test_outcome_survives_restart(env, tmp_path):
    db = tmp_path / "restart.db"
    shutil.copy(env.root / "cura_test.db", db)
    url = f"sqlite:///{db.as_posix()}"
    settings = make_settings(env.root, url, env.data_dir, env.bundle_dir)

    # 1st process: record an outcome through the API, then shut the app (and its engine) down
    with TestClient(create_app(settings)) as c:
        item = c.get("/api/v1/worklist?page_size=1", headers=COORD).json()["items"][0]
        created = c.post(f"/api/v1/episodes/{item['episode_id']}/outreach-attempts", headers=COORD,
                         json={"channel": "phone", "outcome": "reached_already_done",
                               "note": "patient attended elsewhere"}).json()

    # brand-new engine + session: the row is on disk
    engine = make_engine(url)
    with make_session_factory(engine)() as s:
        row = s.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.id == created["id"])).scalar_one()
        assert row.outcome == "reached_already_done" and row.note == "patient attended elsewhere"
        assert row.attempted_at.tzinfo is not None
    engine.dispose()

    # 2nd process: a fresh app still returns it, including in the worklist "last contact"
    with TestClient(create_app(settings)) as c:
        hist = c.get(f"/api/v1/episodes/{item['episode_id']}/outreach-attempts", headers=COORD).json()
        assert hist[0]["id"] == created["id"] and hist[0]["outcome"] == "reached_already_done"
        wl = c.get(f"/api/v1/worklist?q={item['patient']['synthetic_mrn']}", headers=COORD).json()
        assert next(i for i in wl["items"] if i["episode_id"] == item["episode_id"])["last_outreach"]["id"] \
            == created["id"]
