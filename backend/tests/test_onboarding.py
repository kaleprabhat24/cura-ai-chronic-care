"""Patient onboarding: registration, history, duplicates, rollback, roles and point-in-time scoring.

Runs on an isolated copy of the test database (never a live database). An injectable clock lets a test
move "now" forward to a follow-up's prediction time without changing any stored recorded times.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from conftest import ADMIN, ANALYST, COORD, make_settings
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from cura_api import models as m
from cura_api.db import make_engine, make_session_factory
from cura_api.main import create_app
from cura_api.services import onboarding
from cura_api.services.seed import seed_database
from cura_ml.features import spec as S
from cura_ml.modeling.bundle import load_bundle

API = "/api/v1"
REAL_NOW = datetime.now(UTC).replace(microsecond=0)


class Clock:
    def __init__(self) -> None:
        self.now = REAL_NOW

    def __call__(self) -> datetime:
        return self.now


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def day(dt: datetime) -> str:
    return dt.date().isoformat()


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def world(env, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onboarding")
    db = tmp / "onboarding.db"
    shutil.copy(env.root / "cura_test.db", db)
    url = f"sqlite:///{db.as_posix()}"
    clock = Clock()
    settings = make_settings(env.root, url, env.data_dir, env.bundle_dir)
    engine = make_engine(url)
    factory = make_session_factory(engine)
    with factory() as s:
        batch = s.execute(select(m.ScoringRun).where(m.ScoringRun.scope == "eligible")).scalars().first()
        before = {
            "batch_id": batch.id,
            "batch_scores": s.execute(
                select(func.count(), func.sum(m.RiskScore.probability)).where(
                    m.RiskScore.scoring_run_id == batch.id
                )
            ).one(),
            "data_hash": tree_hash(env.data_dir / "records"),
            "bundle_hash": tree_hash(env.bundle_dir),
        }
    with TestClient(create_app(settings, clock=clock), raise_server_exceptions=False) as client:
        yield {
            "client": client,
            "clock": clock,
            "factory": factory,
            "env": env,
            "url": url,
            "settings": settings,
            "before": before,
        }
    engine.dispose()


def registration(**overrides) -> dict:
    """A realistic diabetes + hypertension registration whose next T2D review is ~32 days away."""
    last_review = REAL_NOW - timedelta(days=150)
    body = {
        "submission_id": str(uuid.uuid4()),
        "given_name": "Ada",
        "family_name": "Test" + "".join(chr(97 + int(c, 16)) for c in uuid.uuid4().hex[:8]),
        "birth_date": "1961-04-12",
        "sex": "female",
        "deprivation_quintile": 4,
        "preferred_language": "en",
        "registered_on": day(REAL_NOW - timedelta(days=900)),
        "conditions": [
            {"programme": "t2d", "onset_date": "2015-03-01"},
            {"programme": "htn", "onset_date": "2018-07-01"},
        ],
        "history": {
            "appointments": [
                {
                    "programme": "t2d",
                    "scheduled_for": iso(last_review - timedelta(days=200)),
                    "status": "no_show",
                },
                {"programme": "t2d", "scheduled_for": iso(last_review), "status": "attended"},
                {"programme": "t2d", "scheduled_for": iso(REAL_NOW + timedelta(days=40)), "status": "booked"},
            ],
            "encounters": [
                {
                    "encounter_type": "review_t2d",
                    "start_at": iso(last_review - timedelta(days=182)),
                    "duration_minutes": 20,
                },
                {"encounter_type": "review_t2d", "start_at": iso(last_review), "duration_minutes": 25},
                {"encounter_type": "review_htn", "start_at": iso(REAL_NOW - timedelta(days=100))},
                {"encounter_type": "acute", "start_at": iso(REAL_NOW - timedelta(days=60))},
            ],
            "observations": [
                {"kind": "hba1c", "value": 8.4, "unit": "%", "effective_at": iso(last_review)},
                {
                    "kind": "hba1c",
                    "value": 7.9,
                    "unit": "%",
                    "effective_at": iso(last_review - timedelta(days=182)),
                },
                {
                    "kind": "sbp",
                    "value": 152,
                    "unit": "mm[Hg]",
                    "effective_at": iso(REAL_NOW - timedelta(days=100)),
                },
                {
                    "kind": "dbp",
                    "value": 88,
                    "unit": "mm[Hg]",
                    "effective_at": iso(REAL_NOW - timedelta(days=100)),
                },
                {"kind": "weight", "value": 82.5, "unit": "kg", "effective_at": iso(last_review)},
            ],
            "medications": [
                {"medication": "metformin", "start_date": "2015-04-01"},
                {"medication": "lisinopril", "start_date": "2018-08-01"},
                {"medication": "glipizide", "start_date": "2020-01-01", "end_date": "2021-06-30"},
            ],
            "outreach": [
                {
                    "programme": "t2d",
                    "channel": "phone",
                    "outcome": "no_answer",
                    "attempted_at": iso(last_review - timedelta(days=100)),
                    "next_action": "none",
                },
            ],
        },
    }
    body.update(overrides)
    return body


def post(world, body, headers=COORD):
    return world["client"].post(f"{API}/patients", json=body, headers=headers)


# ------------------------------------------------------------------------------ creation
def test_register_patient_with_history_persists_everything(world):
    world["clock"].now = REAL_NOW
    body = registration()
    r = post(world, body)
    assert r.status_code == 201, r.text
    res = r.json()
    pid = res["patient_id"]
    assert pid.startswith("pt_app_") and res["synthetic_mrn"].startswith("SYNAPP") and res["created"]
    assert res["history_counts"] == {
        "appointments": 3,
        "encounters": 4,
        "observations": 5,
        "medications": 3,
        "outreach": 1,
    }
    with world["factory"]() as s:
        assert s.get(m.Patient, pid).family_name == body["family_name"]
        assert (
            s.execute(
                select(func.count()).select_from(m.Condition).where(m.Condition.patient_id == pid)
            ).scalar_one()
            == 2
        )
        obs = s.execute(select(m.Observation).where(m.Observation.patient_id == pid)).scalars().all()
        assert len(obs) == 5 and {o.unit for o in obs} == {"%", "mm[Hg]", "kg"}
        # effective dates kept as entered; recorded times are the server's entry time (not backdated)
        assert all(o.recorded_at == REAL_NOW for o in obs)
        assert min(o.effective_at for o in obs) < REAL_NOW - timedelta(days=300)
        appts = s.execute(select(m.Appointment).where(m.Appointment.patient_id == pid)).scalars().all()
        assert all(a.booked_at == REAL_NOW and a.status_recorded_at == REAL_NOW for a in appts)
        meds = {
            x.display: x
            for x in s.execute(
                select(m.MedicationStatement).where(m.MedicationStatement.patient_id == pid)
            ).scalars()
        }
        assert meds["glipizide"].end_date.isoformat() == "2021-06-30" and meds["metformin"].end_date is None
        outreach = (
            s.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.patient_id == pid)).scalars().all()
        )
        assert len(outreach) == 1 and outreach[0].origin == "app" and outreach[0].episode_id is not None
        eps = s.execute(select(m.FollowupEpisode).where(m.FollowupEpisode.patient_id == pid)).scalars().all()
        t2d_latest = max((e for e in eps if e.programme == "t2d"), key=lambda e: e.index_event_at)
        last_review = REAL_NOW - timedelta(days=150)
        assert t2d_latest.due_date == last_review.date() + timedelta(days=182)  # existing programme rule
        assert t2d_latest.prediction_at.date() == t2d_latest.due_date - timedelta(days=14)
        audit = (
            s.execute(
                select(m.AuditEvent).where(
                    m.AuditEvent.action == "patient.register", m.AuditEvent.entity_id == pid
                )
            )
            .scalars()
            .all()
        )
        assert len(audit) == 1 and audit[0].actor_id == "coord.test"
        assert "Ada" not in str(audit[0].details)  # no names in the audit details
        assert (
            s.execute(
                select(func.count()).select_from(m.RiskScore).where(m.RiskScore.patient_id == pid)
            ).scalar_one()
            == 0
        )
    t2d = next(x for x in res["score_status"] if x["programme"] == "t2d")
    assert t2d["status"] == "unavailable" and t2d["reason"] == "prediction_time_not_reached"
    assert t2d["probability"] is None and "14 days before" in t2d["message"]
    world["registered"] = pid


def test_minimal_registration_without_history_has_no_score(world):
    world["clock"].now = REAL_NOW
    body = registration(conditions=[{"programme": "htn", "onset_date": "2019-01-01"}], history={})
    r = post(world, body)
    assert r.status_code == 201, r.text
    res = r.json()
    assert res["history_counts"] == {
        "appointments": 0,
        "encounters": 0,
        "observations": 0,
        "medications": 0,
        "outreach": 0,
    }
    assert res["episodes"] == []
    assert res["score_status"] == [
        {**res["score_status"][0], "programme": "htn", "status": "unavailable", "reason": "no_review_history"}
    ]
    assert "365 days later" in res["score_status"][0]["message"]
    pid = res["patient_id"]
    c = world["client"]
    assert c.get(f"{API}/patients/{pid}", headers=COORD).json()["patient"]["id"] == pid
    listed = c.get(f"{API}/patients?origin=registered&q={body['family_name']}", headers=COORD).json()
    assert [i["patient"]["id"] for i in listed["items"]] == [pid] and listed["items"][0]["registered_in_app"]
    assert (
        c.post(f"{API}/patients/{pid}/score", headers=COORD).json()["score_status"][0]["reason"]
        == "no_review_history"
    )


def test_history_entered_after_prediction_time_is_not_backdated(world):
    """Last review 200 days ago: the T2D prediction time (due - 14) passed ~32 days ago, before any of this
    history was recorded. No score may be produced for that follow-up."""
    world["clock"].now = REAL_NOW
    review = REAL_NOW - timedelta(days=200)
    body = registration(
        conditions=[{"programme": "t2d", "onset_date": "2016-01-01"}],
        history={
            "encounters": [{"encounter_type": "review_t2d", "start_at": iso(review)}],
            "observations": [{"kind": "hba1c", "value": 9.1, "unit": "%", "effective_at": iso(review)}],
        },
    )
    res = post(world, body).json()
    st = res["score_status"][0]
    assert st["status"] == "unavailable" and st["reason"] == "history_recorded_after_prediction_time"
    again = world["client"].post(f"{API}/patients/{res['patient_id']}/score", headers=COORD).json()
    assert again["score_status"][0]["status"] == "unavailable"
    with world["factory"]() as s:
        assert (
            s.execute(
                select(func.count())
                .select_from(m.RiskScore)
                .where(m.RiskScore.patient_id == res["patient_id"])
            ).scalar_one()
            == 0
        )


# ------------------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (
            lambda b: b["history"]["observations"].append(
                {
                    "kind": "hba1c",
                    "value": 64,
                    "unit": "mmol/mol",
                    "effective_at": iso(REAL_NOW - timedelta(days=3)),
                }
            ),
            "must be recorded in '%'",
        ),
        (
            lambda b: b["history"]["observations"].append(
                {
                    "kind": "sbp",
                    "value": 400,
                    "unit": "mm[Hg]",
                    "effective_at": iso(REAL_NOW - timedelta(days=3)),
                }
            ),
            "outside the plausible range",
        ),
        (
            lambda b: b["history"]["encounters"].append(
                {"encounter_type": "acute", "start_at": iso(REAL_NOW + timedelta(days=3))}
            ),
            "cannot be in the future",
        ),
        (
            lambda b: b["history"]["appointments"].append(
                {"programme": "t2d", "scheduled_for": iso(REAL_NOW + timedelta(days=3)), "status": "attended"}
            ),
            "future appointment cannot have an attendance result",
        ),
        (
            lambda b: b["history"]["outreach"].append(
                {
                    "programme": "t2d",
                    "channel": "phone",
                    "outcome": "letter_sent",
                    "attempted_at": iso(REAL_NOW),
                }
            ),
            "outcome",
        ),
        (
            lambda b: b.update(conditions=[{"programme": "htn", "onset_date": "2019-01-01"}]),
            "needs that condition",
        ),
        (lambda b: b.update(birth_date="2030-01-01"), "aged 18-110"),
        (
            lambda b: b["history"]["medications"].append(
                {"medication": "metformin", "start_date": "2020-01-01", "end_date": "2019-01-01"}
            ),
            "end_date",
        ),
        (lambda b: b.update(conditions=[]), "conditions"),
        (lambda b: b.update(sex="unknown"), "sex"),
        (lambda b: b.update(given_name="Robert'); DROP TABLE patients;--"), "given_name"),
    ],
)
def test_invalid_registrations_rejected(world, mutate, fragment):
    body = registration()
    mutate(body)
    r = post(world, body)
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert fragment in str(err["details"])
    with world["factory"]() as s:
        assert s.get(m.Patient, onboarding.app_id("pt", body["submission_id"])) is None


# ------------------------------------------------------------------------------ duplicates + rollback
def test_duplicate_submission_and_possible_duplicate_patient(world):
    world["clock"].now = REAL_NOW
    body = registration()
    first, second = post(world, body), post(world, body)
    assert first.status_code == 201 and second.status_code == 200
    assert second.json()["created"] is False and second.json()["patient_id"] == first.json()["patient_id"]
    assert second.json()["history_counts"] == first.json()["history_counts"]
    with world["factory"]() as s:
        assert (
            s.execute(
                select(func.count())
                .select_from(m.Patient)
                .where(m.Patient.family_name == body["family_name"])
            ).scalar_one()
            == 1
        )
        assert (
            s.execute(
                select(func.count())
                .select_from(m.Observation)
                .where(m.Observation.patient_id == first.json()["patient_id"])
            ).scalar_one()
            == 5
        )
    twin = dict(body, submission_id=str(uuid.uuid4()))
    r = post(world, twin)
    assert r.status_code == 409 and r.json()["error"]["code"] == "possible_duplicate"
    assert r.json()["error"]["details"]["patient_id"] == first.json()["patient_id"]
    assert post(world, dict(twin, allow_possible_duplicate=True)).status_code == 201


def test_failed_registration_rolls_back_completely(world, monkeypatch):
    body = registration()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure after the history rows were written")

    monkeypatch.setattr(onboarding, "refresh_episodes", boom)
    r = post(world, body)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    pid = onboarding.app_id("pt", body["submission_id"])
    with world["factory"]() as s:
        assert s.get(m.Patient, pid) is None
        for model in (
            m.Condition,
            m.Appointment,
            m.Encounter,
            m.Observation,
            m.MedicationStatement,
            m.OutreachAttempt,
        ):
            assert (
                s.execute(select(func.count()).select_from(model).where(model.patient_id == pid)).scalar_one()
                == 0
            )
        assert (
            s.execute(
                select(func.count()).select_from(m.AuditEvent).where(m.AuditEvent.entity_id == pid)
            ).scalar_one()
            == 0
        )


# ------------------------------------------------------------------------------ roles
def test_only_coordinators_and_admins_can_register_or_add_history(world):
    c = world["client"]
    body = registration()
    for headers, code in ((ANALYST, 403), ({}, 401)):
        assert post(world, body, headers).status_code == code
        assert c.get(f"{API}/patients", headers=headers).status_code == code
        assert (
            c.post(
                f"{API}/patients/pt_000001/history",
                headers=headers,
                json={"submission_id": str(uuid.uuid4()), "history": {}},
            ).status_code
            == code
        )
        assert c.get(f"{API}/patients/pt_000001/score-status", headers=headers).status_code == code
        assert c.post(f"{API}/patients/pt_000001/score", headers=headers).status_code == code
    assert post(world, registration(), ADMIN).status_code == 201
    with world["factory"]() as s:
        assert s.get(m.Patient, onboarding.app_id("pt", body["submission_id"])) is None


# ------------------------------------------------------------------------------ point-in-time scoring
def test_score_uses_saved_model_at_prediction_time_only(world):
    c, clock, pid = world["client"], world["clock"], world["registered"]
    status = c.get(f"{API}/patients/{pid}/score-status", headers=COORD).json()
    t2d = next(x for x in status["score_status"] if x["programme"] == "t2d")
    assert t2d["reason"] == "prediction_time_not_reached"
    t0 = datetime.fromisoformat(t2d["prediction_at"].replace("Z", "+00:00"))

    # A blood-pressure reading taken before t0 but recorded AFTER t0 must not influence the score.
    # (An HbA1c would complete the diabetes review under the existing rules, so BP is used here.)
    clock.now = t0 + timedelta(days=1)
    late = c.post(
        f"{API}/patients/{pid}/history",
        headers=COORD,
        json={
            "submission_id": str(uuid.uuid4()),
            "history": {
                "observations": [
                    {
                        "kind": "sbp",
                        "value": 199,
                        "unit": "mm[Hg]",
                        "effective_at": iso(REAL_NOW - timedelta(days=1)),
                    }
                ]
            },
        },
    )
    assert late.status_code == 201, late.text
    t2d_now = next(x for x in late.json()["score_status"] if x["programme"] == "t2d")
    assert t2d_now["status"] == "scored", t2d_now
    rid = t2d_now["risk_score_id"]

    bundle = load_bundle(world["env"].bundle_dir)
    with world["factory"]() as s:
        rs = s.get(m.RiskScore, rid)
        run = s.get(m.ScoringRun, rs.scoring_run_id)
        assert run.scope == "patient" and run.model_version == bundle.version == rs.model_version
        assert run.tier_thresholds == bundle.metadata["tier_thresholds"]
        assert run.feature_spec_version == S.FEATURE_SPEC_VERSION
        assert rs.prediction_at == t0 and rs.scored_at == clock.now
        # the feature vector uses only history recorded before t0: the BP entered after t0 is excluded
        assert rs.feature_snapshot["sbp_last"] == pytest.approx(152)
        assert rs.feature_snapshot["sbp_count_24m"] == 1
        assert rs.feature_snapshot["hba1c_last"] == pytest.approx(8.4)
        assert rs.feature_snapshot["hba1c_count_24m"] == 2
        # stored probability, tier and explanation are exactly what the saved model gives for this vector
        frame = pd.DataFrame(
            [
                {
                    "episode_id": rs.episode_id,
                    **{k: (float("nan") if v is None else v) for k, v in rs.feature_snapshot.items()},
                }
            ]
        )
        pred = bundle.predict(frame).iloc[0]
        assert rs.probability == pytest.approx(pred["probability"], abs=1e-12)
        assert rs.risk_tier == pred["risk_tier"]
        expl = bundle.explain(frame, top_k=5)[0]
        assert [f["feature"] for f in rs.explanation["factors"]] == [f["feature"] for f in expl["factors"]]
        assert (
            s.execute(
                select(func.count()).select_from(m.AuditEvent).where(m.AuditEvent.action == "scoring.patient")
            ).scalar_one()
            >= 1
        )

    # visible in the worklist (predictive queue), Patient 360 and the risk-score detail
    mrn = c.get(f"{API}/patients/{pid}", headers=COORD).json()["patient"]["synthetic_mrn"]
    wl = c.get(f"{API}/worklist?q={mrn}&queue=predictive", headers=COORD).json()
    item = next(i for i in wl["items"] if i["episode_id"] == rs.episode_id)
    assert item["prediction"]["risk_score_id"] == rid
    assert item["score_context"]["score_status"] == "predictive"
    assert item["score_context"]["as_of"] == clock.now.date().isoformat()
    detail = c.get(f"{API}/patients/{pid}", headers=COORD).json()
    ep = next(e for e in detail["episodes"] if e["id"] == rs.episode_id)
    assert ep["latest_prediction"]["risk_score_id"] == rid
    rsd = c.get(f"{API}/risk-scores/{rid}", headers=COORD).json()
    assert rsd["is_latest_run"] is True and rsd["explanation"]["factors"]
    # scoring again creates nothing new
    again = c.post(f"{API}/patients/{pid}/score", headers=COORD).json()
    assert next(x for x in again["score_status"] if x["programme"] == "t2d")["status"] == "already_scored"
    with world["factory"]() as s:
        assert (
            s.execute(
                select(func.count()).select_from(m.RiskScore).where(m.RiskScore.episode_id == rs.episode_id)
            ).scalar_one()
            == 1
        )


def test_add_history_refreshes_timeline_and_keeps_scores(world):
    c, pid = world["client"], world["registered"]
    with world["factory"]() as s:
        before = {
            r.id: r.probability
            for r in s.execute(select(m.RiskScore).where(m.RiskScore.patient_id == pid)).scalars()
        }
    body = {
        "submission_id": str(uuid.uuid4()),
        "history": {
            "encounters": [{"encounter_type": "acute", "start_at": iso(REAL_NOW - timedelta(days=2))}],
            "medications": [{"medication": "atorvastatin", "start_date": day(REAL_NOW - timedelta(days=2))}],
        },
    }
    r = c.post(f"{API}/patients/{pid}/history", headers=COORD, json=body)
    assert r.status_code == 201 and r.json()["history_counts"]["encounters"] == 1
    assert c.post(f"{API}/patients/{pid}/history", headers=COORD, json=body).status_code == 200  # replay
    tl = c.get(f"{API}/patients/{pid}/timeline?types=medication_start", headers=COORD).json()
    assert any("atorvastatin" in e["title"] for e in tl["items"])
    with world["factory"]() as s:
        after = {
            r.id: r.probability
            for r in s.execute(select(m.RiskScore).where(m.RiskScore.patient_id == pid)).scalars()
        }
        assert after == before
        assert (
            s.execute(
                select(func.count())
                .select_from(m.MedicationStatement)
                .where(
                    m.MedicationStatement.patient_id == pid, m.MedicationStatement.display == "atorvastatin"
                )
            ).scalar_one()
            == 1
        )
    bad = c.post(
        f"{API}/patients/{pid}/history",
        headers=COORD,
        json={"submission_id": str(uuid.uuid4()), "history": {}},
    )
    assert bad.status_code == 422


def test_history_for_synthetic_patient_does_not_touch_cohort_episodes_or_scores(world):
    c = world["client"]
    with world["factory"]() as s:
        pid = (
            s.execute(select(m.Patient.id).where(~m.Patient.id.like("pt\\_app\\_%", escape="\\")))
            .scalars()
            .first()
        )
        eps = {
            e.id: e.row_hash
            for e in s.execute(select(m.FollowupEpisode).where(m.FollowupEpisode.patient_id == pid)).scalars()
        }
        scores = s.execute(
            select(func.count()).select_from(m.RiskScore).where(m.RiskScore.patient_id == pid)
        ).scalar_one()
    r = c.post(
        f"{API}/patients/{pid}/history",
        headers=COORD,
        json={
            "submission_id": str(uuid.uuid4()),
            "history": {
                "observations": [
                    {
                        "kind": "weight",
                        "value": 77,
                        "unit": "kg",
                        "effective_at": iso(REAL_NOW - timedelta(days=1)),
                    }
                ]
            },
        },
    )
    assert r.status_code == 201 and "synthetic cohort" in " ".join(r.json()["notes"])
    with world["factory"]() as s:
        assert {
            e.id: e.row_hash
            for e in s.execute(select(m.FollowupEpisode).where(m.FollowupEpisode.patient_id == pid)).scalars()
        } == eps
        assert (
            s.execute(
                select(func.count()).select_from(m.RiskScore).where(m.RiskScore.patient_id == pid)
            ).scalar_one()
            == scores
        )
    assert c.post(f"{API}/patients/{pid}/score", headers=COORD).status_code == 409


# ------------------------------------------------------------------------------ persistence + regressions
def test_registered_patient_survives_a_fresh_session_and_app(world):
    pid = world["registered"]
    engine = make_engine(world["url"])
    with make_session_factory(engine)() as s:
        assert s.get(m.Patient, pid) is not None
        assert (
            s.execute(
                select(func.count()).select_from(m.Observation).where(m.Observation.patient_id == pid)
            ).scalar_one()
            >= 5
        )
    engine.dispose()
    with TestClient(create_app(world["settings"])) as fresh:
        assert fresh.get(f"{API}/patients/{pid}", headers=COORD).status_code == 200
        res = fresh.get(f"{API}/patients?origin=registered&page_size=100", headers=COORD).json()
        assert pid in {i["patient"]["id"] for i in res["items"]}


def test_existing_data_model_and_cohort_scores_unchanged(world):
    b = world["before"]
    env = world["env"]
    assert tree_hash(env.data_dir / "records") == b["data_hash"]
    assert tree_hash(env.bundle_dir) == b["bundle_hash"]
    c = world["client"]
    with world["factory"]() as s:
        assert (
            s.execute(
                select(func.count(), func.sum(m.RiskScore.probability)).where(
                    m.RiskScore.scoring_run_id == b["batch_id"]
                )
            ).one()
            == b["batch_scores"]
        )
    dash = c.get(f"{API}/dashboard/summary", headers=COORD).json()
    assert (
        dash["latest_scoring_run"]["id"] == b["batch_id"]
        and dash["latest_scoring_run"]["scope"] == "eligible"
    )
    assert c.get(f"{API}/health").json()["latest_scoring_run_id"] == b["batch_id"]
    dq = c.get(f"{API}/data-quality", headers=ANALYST).json()
    assert dq["status"] == "ok", [x for x in dq["checks"] if not x["ok"]]
    assert dq["table_counts"]["patients"]["entered_in_app"] >= 3
    # re-seeding the synthetic dataset ignores rows entered in the app
    again = seed_database(world["factory"], env.data_dir, actor="system:test")
    assert all(v["inserted"] == v["updated"] == v["not_in_source"] == 0 for v in again["tables"].values())
    with world["factory"]() as s:
        assert s.get(m.Patient, world["registered"]) is not None
    fhir = c.get(f"{API}/patients/{world['registered']}/fhir", headers=COORD).json()
    assert any(e["resource"]["resourceType"] == "RiskAssessment" for e in fhir["entry"])
