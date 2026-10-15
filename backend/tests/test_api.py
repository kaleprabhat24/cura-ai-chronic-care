"""REST API: auth, pagination, filtering, validation, explanations, outreach, aggregates, OpenAPI."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from conftest import ADMIN, ANALYST, COORD, make_settings
from fastapi.testclient import TestClient
from sqlalchemy import select

from cura_api import models as m
from cura_api.main import create_app

API = "/api/v1"


def _error(r, status: int, code: str) -> dict:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code
    return body["error"]


# ------------------------------------------------------------------------------ system + auth
def test_health(client):
    j = client.get(f"{API}/health").json()
    assert j["status"] == "ok" and j["database"] == "ok" and j["dataset_loaded"]
    assert j["auth_mode"] == "dev" and j["production_auth_implemented"] is False


def test_authentication_required_and_roles_enforced(client):
    _error(client.get(f"{API}/worklist"), 401, "unauthenticated")
    _error(client.get(f"{API}/worklist", headers={"X-Dev-User": "x", "X-Dev-Role": "superuser"}),
           401, "unauthenticated")
    _error(client.get(f"{API}/worklist", headers=ANALYST), 403, "forbidden")
    _error(client.get(f"{API}/audit-events", headers=COORD), 403, "forbidden")
    _error(client.get(f"{API}/data-quality", headers=COORD), 403, "forbidden")
    assert client.get(f"{API}/dashboard/summary", headers=ANALYST).status_code == 200
    assert client.get(f"{API}/data-quality", headers=ANALYST).status_code == 200
    assert client.get(f"{API}/audit-events", headers=ADMIN).status_code == 200
    assert client.get(f"{API}/me", headers=COORD).json() == {
        "user_id": "coord.test", "role": "coordinator", "auth_mode": "dev", "display_name": None}


def test_dev_auth_refuses_production_and_supabase_requires_configuration(env):
    with pytest.raises(RuntimeError, match="refuses to run"):
        create_app(make_settings(env.root, env.database_url, env.data_dir, env.bundle_dir,
                                 environment="production"))
    with pytest.raises(RuntimeError, match="requires SUPABASE_URL"):
        create_app(make_settings(env.root, env.database_url, env.data_dir, env.bundle_dir,
                                 auth_mode="supabase"))


# ------------------------------------------------------------------------------ worklist
def _all_items(client, query: str = "") -> list[dict]:
    items, page = [], 1
    while True:
        j = client.get(f"{API}/worklist?page={page}&page_size=40{query}", headers=COORD).json()
        items += j["items"]
        if page >= j["total_pages"]:
            return items
        page += 1


def test_worklist_pagination_is_complete_and_disjoint(client):
    first = client.get(f"{API}/worklist?page_size=40", headers=COORD).json()
    assert first["total"] > 40 and first["page"] == 1 and first["total_pages"] == -(-first["total"] // 40)
    items = _all_items(client)
    ids = [i["episode_id"] for i in items]
    assert len(ids) == len(set(ids)) == first["total"]
    beyond = client.get(f"{API}/worklist?page={first['total_pages'] + 1}&page_size=40", headers=COORD).json()
    assert beyond["items"] == [] and beyond["total"] == first["total"]


def test_worklist_matches_open_episodes_in_database(env, client):
    engine, factory = env.factory()
    with factory() as s:
        rows = s.execute(select(m.FollowupEpisode.id).join(m.EpisodeOutcome)
                         .join(m.Patient, m.Patient.id == m.FollowupEpisode.patient_id)
                         .where(m.FollowupEpisode.eligible.is_(True), m.EpisodeOutcome.completed_at.is_(None),
                                m.EpisodeOutcome.censored.is_(False), m.Patient.death_date.is_(None),
                                m.Patient.transfer_out_date.is_(None))).scalars().all()
    engine.dispose()
    assert {i["episode_id"] for i in _all_items(client)} == set(rows)
    dash = client.get(f"{API}/dashboard/summary", headers=COORD).json()
    assert dash["open_episodes"] == len(rows)
    assert sum(dash["open_by_tier"].values()) + dash["open_unscored"] == len(rows)


def test_worklist_sorting(client):
    def values(sort: str, key, order: str = "") -> list:
        j = client.get(f"{API}/worklist?page_size=100&sort={sort}{order}", headers=COORD).json()
        return [key(i) for i in j["items"]]
    ranks = values("priority", lambda i: i["prediction"]["rank"])
    assert ranks == sorted(ranks)
    probs = values("probability", lambda i: i["prediction"]["probability"])
    assert probs == sorted(probs, reverse=True)
    due = values("due_date", lambda i: i["due_date"], "&order=desc")
    assert due == sorted(due, reverse=True)
    names = values("patient_name", lambda i: i["patient"]["display_name"].split(" ", 1)[1])
    assert names == sorted(names)


def test_worklist_filters_and_search(client):
    high = _all_items(client, "&tier=high")
    assert high and all(i["prediction"]["risk_tier"] == "high" for i in high)
    two = _all_items(client, "&tier=high&tier=medium&condition=htn")
    assert all(i["prediction"]["risk_tier"] in ("high", "medium") and i["programme"] == "htn" for i in two)
    window = _all_items(client, "&due_from=2024-06-01&due_to=2024-12-31")
    assert window and all("2024-06-01" <= i["due_date"] <= "2024-12-31" for i in window)
    od = _all_items(client, "&overdue=false")
    assert all(i["days_until_due"] >= 0 for i in od)
    target = high[0]["patient"]
    by_mrn = client.get(f"{API}/worklist?q={target['synthetic_mrn'].lower()}", headers=COORD).json()
    assert {i["patient"]["id"] for i in by_mrn["items"]} == {target["id"]}
    by_name = client.get(f"{API}/worklist?q={target['display_name'].split()[1]}&page_size=100",
                         headers=COORD).json()
    assert target["id"] in {i["patient"]["id"] for i in by_name["items"]}
    assert client.get(f"{API}/worklist?q=%25%25", headers=COORD).json()["total"] == 0  # literal '%'


@pytest.mark.parametrize("query", ["page_size=1000", "page=0", "sort=risk", "tier=extreme",
                                   "condition=copd", "due_from=2024-02-30", "order=up",
                                   "due_from=2024-05-01&due_to=2024-01-01"])
def test_worklist_invalid_requests(client, query):
    _error(client.get(f"{API}/worklist?{query}", headers=COORD), 422, "validation_error")


# ------------------------------------------------------------------------------ patient + scores
def test_patient_profile_timeline_and_404(env, client):
    item = client.get(f"{API}/worklist?page_size=1", headers=COORD).json()["items"][0]
    pid = item["patient"]["id"]
    p = client.get(f"{API}/patients/{pid}", headers=COORD).json()
    assert p["patient"]["id"] == pid and p["patient"]["is_synthetic"]
    assert p["patient"]["provenance"] == "recorded_synthetic_clinical"
    assert item["episode_id"] in {e["id"] for e in p["episodes"]}
    ep = next(e for e in p["episodes"] if e["id"] == item["episode_id"])
    assert ep["latest_prediction"]["risk_score_id"] == item["prediction"]["risk_score_id"]
    assert ep["recorded_outcome"]["provenance"] == "recorded_synthetic_outcome"
    tl = client.get(f"{API}/patients/{pid}/timeline?limit=1000", headers=COORD).json()
    types = {e["type"] for e in tl["items"]}
    assert {"appointment", "encounter", "observation", "risk_score"} <= types
    prov = {e["type"]: e["provenance"] for e in tl["items"]}
    assert prov["risk_score"] == "simulated_prediction" and prov["encounter"] == "recorded_synthetic_clinical"
    times = [e["occurred_at"] for e in tl["items"]]
    assert times == sorted(times, reverse=True)
    only = client.get(f"{API}/patients/{pid}/timeline?types=observation", headers=COORD).json()
    assert {e["type"] for e in only["items"]} == {"observation"}
    engine, factory = env.factory()
    with factory() as s:
        n_obs = len(s.execute(select(m.Observation.id).where(m.Observation.patient_id == pid)).all())
    engine.dispose()
    assert only["total"] == n_obs
    _error(client.get(f"{API}/patients/pt_999999", headers=COORD), 404, "not_found")
    _error(client.get(f"{API}/patients/{pid}/timeline?types=xray", headers=COORD), 422, "validation_error")


def test_risk_score_detail_is_the_stored_explanation(env, client):
    item = client.get(f"{API}/worklist?page_size=1", headers=COORD).json()["items"][0]
    rid = item["prediction"]["risk_score_id"]
    d = client.get(f"{API}/risk-scores/{rid}", headers=COORD).json()
    engine, factory = env.factory()
    with factory() as s:
        r = s.get(m.RiskScore, rid)
        assert d["probability"] == r.probability and d["risk_tier"] == r.risk_tier
        stored = [f["feature"] for f in r.explanation["factors"]]
        assert [f["feature"] for f in d["explanation"]["factors"]] == stored
    engine.dispose()
    assert d["provenance"] == "simulated_prediction"
    assert "not percentages" in d["explanation"]["units"]
    assert d["run"]["model_version"] == item["prediction"]["model_version"]
    assert len(d["features"]) == 44 and all("label" in f for f in d["features"])
    hist = client.get(f"{API}/episodes/{item['episode_id']}/risk-scores", headers=COORD).json()
    assert rid in {h["id"] for h in hist}
    _error(client.get(f"{API}/risk-scores/99999999", headers=COORD), 404, "not_found")


# ------------------------------------------------------------------------------ outreach
def _open_episode(client) -> dict:
    return client.get(f"{API}/worklist?page_size=1&sort=probability", headers=COORD).json()["items"][0]


def test_record_outreach_persists_and_is_audited(env, client):
    item = _open_episode(client)
    eid = item["episode_id"]
    when = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    due = (date.today() + timedelta(days=3)).isoformat()
    r = client.post(f"{API}/episodes/{eid}/outreach-attempts", headers=COORD,
                    json={"channel": "phone", "outcome": "no_answer", "attempted_at": when,
                          "note": "  rang twice  ", "next_action": "retry", "next_action_due": due})
    assert r.status_code == 201, r.text
    o = r.json()
    assert o["origin"] == "app" and o["recorded_by"] == "coord.test" and o["note"] == "rang twice"
    assert o["provenance"] == "recorded_outreach_outcome" and o["next_action_due"] == due
    hist = client.get(f"{API}/episodes/{eid}/outreach-attempts", headers=COORD).json()
    assert hist[0]["id"] == o["id"]
    wl = client.get(f"{API}/worklist?q={item['patient']['synthetic_mrn']}", headers=COORD).json()
    row = next(i for i in wl["items"] if i["episode_id"] == eid)
    assert row["last_outreach"]["id"] == o["id"] and row["outreach_attempts"] == item["outreach_attempts"] + 1
    audit = client.get(f"{API}/audit-events?action=outreach.create", headers=ADMIN).json()
    assert any(e["entity_id"] == o["id"] and e["actor_id"] == "coord.test" for e in audit["items"])
    summary = client.get(f"{API}/outreach/outcomes", headers=ANALYST).json()
    assert summary["attempts_by_outcome"]["app"]["no_answer"] >= 1


@pytest.mark.parametrize("outcome", ["no_answer", "left_message", "reached_booked", "reached_declined",
                                     "reached_already_done", "wrong_number", "other"])
def test_every_outcome_category_accepted(client, outcome):
    eid = _open_episode(client)["episode_id"]
    r = client.post(f"{API}/episodes/{eid}/outreach-attempts", headers=COORD,
                    json={"channel": "sms", "outcome": outcome})
    assert r.status_code == 201 and r.json()["outcome"] == outcome


@pytest.mark.parametrize("body", [
    {"channel": "phone", "outcome": "letter_sent"},
    {"channel": "fax", "outcome": "no_answer"},
    {"channel": "phone", "outcome": "no_answer", "next_action": "retry"},
    {"channel": "phone", "outcome": "no_answer", "next_action": "none", "next_action_due": "2030-01-01"},
    {"channel": "phone", "outcome": "no_answer", "next_action": "retry",
     "attempted_at": "2024-06-10T10:00:00Z", "next_action_due": "2024-06-01"},
    {"channel": "phone", "outcome": "no_answer", "attempted_at": "2999-01-01T10:00:00Z"},
    {"channel": "phone", "outcome": "no_answer", "attempted_at": "2024-06-10T10:00:00"},
    {"channel": "phone", "outcome": "no_answer", "note": "x" * 2001},
    {"channel": "phone", "outcome": "no_answer", "probability": 0.1},
])
def test_invalid_outreach_rejected(client, body):
    eid = _open_episode(client)["episode_id"]
    _error(client.post(f"{API}/episodes/{eid}/outreach-attempts", headers=COORD, json=body),
           422, "validation_error")


def test_outreach_permissions_and_conflicts(env, client):
    eid = _open_episode(client)["episode_id"]
    body = {"channel": "phone", "outcome": "no_answer"}

    def post(episode_id: str, headers: dict | None):
        return client.post(f"{API}/episodes/{episode_id}/outreach-attempts", headers=headers, json=body)

    _error(post(eid, ANALYST), 403, "forbidden")
    _error(post(eid, None), 401, "unauthenticated")
    _error(post("ep_missing", COORD), 404, "not_found")
    engine, factory = env.factory()
    with factory() as s:
        inel = s.execute(select(m.FollowupEpisode.id)
                         .where(m.FollowupEpisode.eligible.is_(False))).scalars().first()
    engine.dispose()
    _error(post(inel, ADMIN), 409, "conflict")


# ------------------------------------------------------------------------------ aggregates + model
def test_outcomes_model_scoring_runs_and_data_quality(env, client):
    o = client.get(f"{API}/outreach/outcomes", headers=ANALYST).json()
    assert o["attempts_by_origin"]["historical_simulation"] > 0
    years = o["completion_by_year"]
    assert years and all(0 <= g["completed_on_time_rate"] <= 1 for g in years)
    mi = client.get(f"{API}/models/current", headers=COORD).json()
    assert mi["model_version"] == "test-api" and mi["synthetic_data_only"]
    assert set(mi["test_metrics"]) == {"prevalence_baseline", "attendance_heuristic",
                                       "logistic_regression", "xgboost"}
    assert client.get(f"{API}/models/test-api", headers=ANALYST).status_code == 200
    _error(client.get(f"{API}/models/nope", headers=ANALYST), 404, "not_found")
    _error(client.get(f"{API}/models/..%5C..%5Csecret", headers=ANALYST), 422, "validation_error")
    runs = client.get(f"{API}/scoring-runs", headers=ANALYST).json()
    assert runs["total"] >= 1 and runs["items"][0]["status"] == "succeeded"
    one = client.get(f"{API}/scoring-runs/{runs['items'][-1]['id']}", headers=ANALYST).json()
    assert one["tier_thresholds"] == mi["tier_thresholds"]
    dq = client.get(f"{API}/data-quality", headers=ANALYST).json()
    assert dq["status"] == "ok", [c for c in dq["checks"] if not c["ok"]]
    assert dq["table_counts"]["patients"]["database"] == dq["table_counts"]["patients"]["dataset_files"]


def test_openapi_documents_the_api(client):
    doc = client.get("/openapi.json").json()
    for path in ("/api/v1/health", "/api/v1/dashboard/summary", "/api/v1/worklist",
                 "/api/v1/patients/{patient_id}", "/api/v1/patients/{patient_id}/timeline",
                 "/api/v1/risk-scores/{risk_score_id}", "/api/v1/episodes/{episode_id}/outreach-attempts",
                 "/api/v1/outreach/outcomes", "/api/v1/models/current", "/api/v1/scoring-runs",
                 "/api/v1/data-quality", "/api/v1/audit-events"):
        assert path in doc["paths"], path
    post = doc["paths"]["/api/v1/episodes/{episode_id}/outreach-attempts"]["post"]
    assert "201" in post["responses"] and "422" in post["responses"]
    params = {p["name"] for p in doc["paths"]["/api/v1/worklist"]["get"]["parameters"]}
    assert {"page", "page_size", "sort", "q", "tier", "condition", "due_from", "due_to",
            "x-dev-user", "x-dev-role"} <= params
    assert "OutreachCreate" in doc["components"]["schemas"]


def test_fresh_client_on_same_app_settings(env):
    with TestClient(create_app(env.settings)) as c:
        assert c.get(f"{API}/health").status_code == 200
