"""Stale-score handling: predictive risk vs historical scores vs overdue (observed) outcomes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from conftest import COORD
from sqlalchemy import select

from cura_api import models as m
from cura_api.api.common import score_context

AS_OF = date(2024, 12, 31)


def _ep(due: date, eligible=True, cold=False) -> m.FollowupEpisode:
    return m.FollowupEpisode(id="ep", patient_id="pt", programme="t2d", due_date=due,
                             window_end_date=due + timedelta(days=30), eligible=eligible,
                             ineligible_reason=None if eligible else "prediction_after_cutoff",
                             is_cold_start=cold,
                             prediction_at=datetime.combine(due - timedelta(days=14), datetime.min.time(),
                                                            tzinfo=UTC))


def _out(completed: datetime | None = None, censored=False, mature=False) -> m.EpisodeOutcome:
    return m.EpisodeOutcome(episode_id="ep", completed_at=completed, censored=censored, mature=mature,
                            completed_in_window=None, label_status="immature")


def _rs(ep: m.FollowupEpisode) -> m.RiskScore:
    return m.RiskScore(id=1, prediction_at=ep.prediction_at, probability=0.7, risk_tier="high")


@pytest.mark.parametrize("due,out,expected_fu,expected_score", [
    (date(2025, 1, 20), _out(), "upcoming", "predictive"),
    (date(2024, 12, 10), _out(), "due_in_grace_window", "predictive"),        # window ends 2025-01-09
    (date(2024, 12, 1), _out(), "due_in_grace_window", "predictive"),         # window ends exactly at as_of
    (date(2024, 11, 30), _out(), "overdue_unresolved", "historical"),         # window ended yesterday
    (date(2021, 3, 1), _out(mature=True), "overdue_unresolved", "historical"),  # years old
    (date(2024, 6, 1), _out(completed=datetime(2024, 6, 3, tzinfo=UTC), mature=True), "completed",
     "historical"),
    (date(2024, 12, 20), _out(completed=datetime(2024, 12, 5, tzinfo=UTC)), "completed", "historical"),
    (date(2024, 6, 1), _out(censored=True, mature=True), "censored", "historical"),
])
def test_score_context_rules(due, out, expected_fu, expected_score):
    ep = _ep(due)
    ctx = score_context(ep, out, _rs(ep), AS_OF)
    assert (ctx.follow_up_status, ctx.score_status) == (expected_fu, expected_score)
    assert ctx.outcome_window_ended == (ep.window_end_date < AS_OF)
    assert ctx.days_overdue == max(0, (AS_OF - due).days)
    assert ctx.score_age_days == (AS_OF - ep.prediction_at.date()).days
    if expected_score == "historical":
        assert "historical, not a current prediction" in ctx.summary
    if expected_fu == "overdue_unresolved":
        assert "Recorded outcome: not completed" in ctx.summary


def test_no_score_and_limited_history():
    ep = _ep(date(2025, 1, 20), cold=True)
    ctx = score_context(ep, _out(), None, AS_OF)
    assert ctx.score_status == "no_score" and ctx.score_age_days is None
    assert ctx.data_sufficiency == "limited_history" and "Limited history" in ctx.summary


def _all(client, query: str) -> list[dict]:
    items, page = [], 1
    while True:
        j = client.get(f"/api/v1/worklist?page={page}&page_size=100{query}", headers=COORD).json()
        items += j["items"]
        if page >= j["total_pages"]:
            return items
        page += 1


def test_worklist_queues_separate_predictive_risk_from_overdue_cases(client):
    everything = _all(client, "")
    predictive = _all(client, "&queue=predictive")
    overdue = _all(client, "&queue=overdue")
    no_score = _all(client, "&queue=no_score")
    assert predictive and overdue
    assert all(i["score_context"]["score_status"] == "predictive" and i["prediction"] for i in predictive)
    live = ("upcoming", "due_in_grace_window")
    assert all(i["score_context"]["follow_up_status"] in live for i in predictive)
    assert all(i["score_context"]["follow_up_status"] == "overdue_unresolved" for i in overdue)
    assert all(i["score_context"]["score_status"] in ("historical", "no_score") for i in overdue)
    assert all(i["score_context"]["outcome_window_ended"] for i in overdue)
    assert all(i["prediction"] is None for i in no_score)
    ids = lambda xs: {i["episode_id"] for i in xs}  # noqa: E731
    assert not ids(predictive) & ids(overdue)
    assert ids(predictive) | ids(overdue) | ids(no_score) == ids(everything)
    # nothing in the full list presents a matured, unresolved episode as a predictive score
    for i in everything:
        ctx = i["score_context"]
        if ctx["outcome_window_ended"]:
            assert ctx["score_status"] != "predictive"
        assert ctx["as_of"] == "2024-12-31"
        if i["prediction"]:
            assert i["prediction"]["prediction_at"] and i["prediction"]["scored_at"]
            assert i["prediction"]["model_version"]


def test_dashboard_counts_are_consistent_with_queues(client):
    d = client.get("/api/v1/dashboard/summary", headers=COORD).json()
    assert d["open_upcoming"] + d["open_in_grace_window"] + d["open_overdue_unresolved"] == d["open_episodes"]
    predictive = client.get("/api/v1/worklist?queue=predictive&page_size=1", headers=COORD).json()["total"]
    overdue = client.get("/api/v1/worklist?queue=overdue&page_size=1", headers=COORD).json()["total"]
    assert sum(d["predictive_by_tier"].values()) == predictive
    assert d["open_overdue_unresolved"] == overdue


def test_historical_scores_keep_provenance_in_detail_and_timeline(env, client):
    url = "/api/v1/worklist?queue=overdue&page_size=1&sort=probability"
    item = client.get(url, headers=COORD).json()["items"][0]
    rid = item["prediction"]["risk_score_id"]
    d = client.get(f"/api/v1/risk-scores/{rid}", headers=COORD).json()
    assert d["score_context"]["score_status"] == "historical"
    assert d["is_latest_run"] is True and d["prediction_at"] and d["scored_at"] and d["model_version"]
    assert all(f["window"] for f in d["explanation"]["factors"])
    tl = client.get(f"/api/v1/patients/{item['patient']['id']}/timeline?types=risk_score",
                    headers=COORD).json()
    assert any(e["detail"]["risk_score_id"] == rid and e["provenance"] == "simulated_prediction"
               for e in tl["items"])
    engine, factory = env.factory()
    with factory() as s:
        assert s.execute(select(m.RiskScore).where(m.RiskScore.id == rid)).scalar_one().probability == \
            pytest.approx(d["probability"])
    engine.dispose()


def test_invalid_queue_rejected(client):
    r = client.get("/api/v1/worklist?queue=stale", headers=COORD)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
