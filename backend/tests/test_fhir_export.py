"""FHIR R4-oriented export: content, references, provenance, structural and R4B-model validation."""

from __future__ import annotations

from collections import Counter

from conftest import ADMIN, ANALYST, COORD
from sqlalchemy import func, select

from cura_api import models as m
from cura_api.services.fhir import r4b_parse_errors, structural_problems

EXPECTED = {"Patient", "Condition", "Encounter", "Appointment", "Observation", "MedicationStatement",
            "RiskAssessment", "Communication"}


def _patients(env, n: int = 8) -> list[str]:
    engine, factory = env.factory()
    with factory() as s:
        ids = s.execute(select(m.RiskScore.patient_id).distinct().order_by(m.RiskScore.patient_id)
                        .limit(n)).scalars().all()
    engine.dispose()
    return ids


def test_bundle_contains_all_resource_types_and_validates(env, client):
    seen: set[str] = set()
    for pid in _patients(env):
        r = client.get(f"/api/v1/patients/{pid}/fhir", headers=COORD)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/fhir+json")
        assert "attachment" in r.headers["content-disposition"]
        bundle = r.json()
        assert bundle["resourceType"] == "Bundle" and bundle["type"] == "collection"
        assert structural_problems(bundle) == []
        assert r4b_parse_errors(bundle) == []
        seen |= {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert seen == EXPECTED


def test_bundle_matches_database_records(env, client):
    pid = _patients(env, 1)[0]
    bundle = client.get(f"/api/v1/patients/{pid}/fhir", headers=COORD).json()
    by_type = Counter(e["resource"]["resourceType"] for e in bundle["entry"])
    engine, factory = env.factory()
    with factory() as s:
        def n(model):
            q = select(func.count()).select_from(model).where(model.patient_id == pid)
            return s.execute(q).scalar_one()
        assert by_type["Patient"] == 1
        assert by_type["Observation"] == n(m.Observation)
        assert by_type["Appointment"] == n(m.Appointment)
        assert by_type["Encounter"] == n(m.Encounter)
        assert by_type["MedicationStatement"] == n(m.MedicationStatement)
        assert by_type["Communication"] == n(m.OutreachAttempt)
        run_id = s.execute(select(func.max(m.ScoringRun.id))).scalar_one()
        scores = {str(r.id): r for r in s.execute(select(m.RiskScore).where(
            m.RiskScore.patient_id == pid, m.RiskScore.scoring_run_id == run_id)).scalars()}
        obs = {o.id.replace("_", "-"): o for o in s.execute(select(m.Observation).where(
            m.Observation.patient_id == pid)).scalars()}
    engine.dispose()
    assert by_type["RiskAssessment"] == len(scores)
    for e in bundle["entry"]:
        res = e["resource"]
        assert {"system": "urn:cura-ai:tags", "code": "synthetic",
                "display": "Synthetic demonstration data"} in res["meta"]["tag"]
        if res["resourceType"] == "RiskAssessment":
            rs = scores[res["identifier"][0]["value"]]
            assert res["method"]["coding"][0]["code"] == rs.model_version
            assert res["prediction"][0]["probabilityDecimal"] == round(rs.probability, 6)
            assert "SIMULATED PREDICTION" in res["note"][0]["text"]
            assert res["subject"]["reference"] == bundle["entry"][0]["fullUrl"]
        if res["resourceType"] == "Observation":
            o = obs[res["id"]]
            assert res["valueQuantity"]["value"] == o.value_num
            assert res["valueQuantity"]["code"] == o.unit and res["valueQuantity"]["system"] == "http://unitsofmeasure.org"


def test_app_outreach_appears_as_communication_and_export_is_audited(env, client):
    item = client.get("/api/v1/worklist?page_size=1&queue=predictive", headers=COORD).json()["items"][0]
    created = client.post(f"/api/v1/episodes/{item['episode_id']}/outreach-attempts", headers=COORD,
                          json={"channel": "phone", "outcome": "wrong_number",
                                "next_action": "update_contact_details",
                                "next_action_due": "2030-01-01"}).json()
    bundle = client.get(f"/api/v1/patients/{item['patient']['id']}/fhir", headers=COORD).json()
    comm = [e["resource"] for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Communication"
            and e["resource"]["identifier"][0]["value"] == created["id"]]
    assert len(comm) == 1 and comm[0]["status"] == "not-done"
    assert "update contact details" in comm[0]["payload"][0]["contentString"]
    audit = client.get("/api/v1/audit-events?action=fhir.export", headers=ADMIN).json()
    assert any(a["entity_id"] == item["patient"]["id"] for a in audit["items"])


def test_fhir_export_permissions_and_404(client):
    assert client.get("/api/v1/patients/pt_000001/fhir", headers=ANALYST).status_code == 403
    assert client.get("/api/v1/patients/pt_000001/fhir").status_code == 401
    r = client.get("/api/v1/patients/pt_404404/fhir", headers=COORD)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_structural_checker_detects_problems():
    bad = {"resourceType": "Bundle", "type": "collection", "entry": [
        {"fullUrl": "urn:uuid:not-valid", "resource": {"resourceType": "Observation", "id": "a_b",
                                                       "valueQuantity": {"value": 1, "unit": "%"},
                                                       "subject": {"reference": "urn:uuid:missing"}}},
        {"fullUrl": "urn:uuid:00000000-0000-0000-0000-000000000001",
         "resource": {"resourceType": "Appointment", "id": "x", "status": "booked",
                      "start": "2024-01-01T10:00:00Z"}},
    ]}
    problems = " | ".join(structural_problems(bad))
    for expected in ("fullUrl not a urn:uuid", "invalid id", "UCUM", "app-3", "app-2", "participant",
                     "unresolved reference"):
        assert expected in problems
