"""FHIR R4-oriented export of one synthetic patient as a `collection` Bundle.

Resources: Patient, Condition, Encounter, Appointment, Observation, MedicationStatement,
RiskAssessment (simulated predictions with model provenance) and Communication (outreach attempts).

* Resource ids are the database ids made FHIR-safe (`_` -> `-`). Each entry's `fullUrl` is a
  deterministic `urn:uuid:` (UUIDv5 of type + id). All references point at those fullUrls, so the
  Bundle is self-contained.
* Every resource is tagged as synthetic (`meta.tag`); risk assessments say they are simulated.
* Validation: `structural_problems()` checks the R4 rules this export relies on, and
  `fhir.resources` R4B models parse every resource (R4B is the closest FHIR release that library
  ships). **Neither is formal R4 conformance**; that needs the official HL7 validator (see
  docs/FHIR_EXPORT.md).
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cura_api import models as m
from cura_api.api.common import effective_scores, score_context
from cura_ml.synth import catalog as cat

NS = uuid.UUID("6f1c2f9e-1d7a-4c3b-9a43-6f7a2e0c5b11")
TAG = {"system": "urn:cura-ai:tags", "code": "synthetic", "display": "Synthetic demonstration data"}
UCUM = "http://unitsofmeasure.org"
FHIR_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
NOMINAL_SLOT_MINUTES = 20
LAB_CODES = {cat.OBSERVATIONS["hba1c"][0]}
APPOINTMENT_STATUS = {"booked": "booked", "attended": "fulfilled", "no_show": "noshow",
                      "cancelled_patient": "cancelled", "cancelled_clinic": "cancelled"}
PARTICIPATION_MODE = {"phone": ("PHONE", "telephone"), "letter": ("MAILWRIT", "mail"),
                      "email": ("EMAILWRIT", "email"), "in_person": ("FACE", "face-to-face")}
NOT_DONE = {"no_answer", "wrong_number"}


def fhir_id(raw: str) -> str:
    out = raw.replace("_", "-")
    return out if FHIR_ID.match(out) else uuid.uuid5(NS, raw).hex


def _full_url(rtype: str, raw_id: str) -> str:
    return f"urn:uuid:{uuid.uuid5(NS, f'{rtype}/{raw_id}')}"


def _ts(v: datetime | None) -> str | None:
    return None if v is None else v.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _d(v: date | None) -> str | None:
    return None if v is None else v.isoformat()


def _meta() -> dict[str, Any]:
    return {"tag": [TAG]}


def _clean(obj: Any) -> Any:
    """Drop None / empty values (FHIR forbids empty elements)."""
    if isinstance(obj, dict):
        cleaned = {k: _clean(v) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
    if isinstance(obj, list):
        return [x for x in (_clean(v) for v in obj) if x not in (None, [], {}, "")]
    return obj


def _payload_text(o: m.OutreachAttempt) -> str:
    text = f"Outcome: {o.outcome.replace('_', ' ')}"
    if o.note:
        text += f". Note: {o.note.rstrip('.')}"
    if o.next_action != "none":
        text += f". Next action: {o.next_action.replace('_', ' ')}"
        if o.next_action_due:
            text += f" by {o.next_action_due.isoformat()}"
    return text


def build_bundle(session: Session, patient_id: str, as_of: date) -> dict[str, Any]:
    p = session.get(m.Patient, patient_id)
    if p is None:
        raise KeyError(patient_id)
    entries: list[dict[str, Any]] = []
    pref = {"reference": _full_url("Patient", p.id)}

    def add(rtype: str, raw_id: str, resource: dict[str, Any]) -> None:
        resource = _clean({"resourceType": rtype, "id": fhir_id(raw_id), "meta": _meta(), **resource})
        entries.append({"fullUrl": _full_url(rtype, raw_id), "resource": resource})

    add("Patient", p.id, {
        "identifier": [{"system": "urn:cura-ai:synthetic-mrn", "value": p.synthetic_mrn}],
        "active": p.death_date is None and p.transfer_out_date is None,
        "name": [{"use": "official", "family": p.family_name, "given": [p.given_name]}],
        "gender": p.sex if p.sex in ("male", "female") else "unknown",
        "birthDate": _d(p.birth_date),
        "deceasedDateTime": _ts(datetime.combine(p.death_date, datetime.min.time(), tzinfo=UTC))
        if p.death_date else None,
        "communication": [{"language": {"coding": [{"system": "urn:ietf:bcp:47",
                                                     "code": p.preferred_language}]}}],
    })
    for c in session.execute(select(m.Condition).where(m.Condition.patient_id == p.id)).scalars():
        add("Condition", c.id, {
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                      "code": "problem-list-item"}]}],
            "code": {"coding": [{"system": c.code_system, "code": c.code, "display": c.display}],
                     "text": c.display},
            "subject": pref, "onsetDateTime": _d(c.onset_date), "recordedDate": _ts(c.recorded_at),
        })
    for a in session.execute(select(m.Appointment).where(m.Appointment.patient_id == p.id)).scalars():
        status = APPOINTMENT_STATUS[a.status]
        add("Appointment", a.id, {
            "status": status,
            "appointmentType": {"text": a.appointment_type.replace("_", " ")},
            "description": f"{a.appointment_type.replace('_', ' ')} (booking source: {a.booking_source})",
            "start": _ts(a.scheduled_for),
            "end": _ts(a.scheduled_for + timedelta(minutes=NOMINAL_SLOT_MINUTES)),
            "minutesDuration": NOMINAL_SLOT_MINUTES,
            "created": _d(a.booked_at.date()),
            "comment": "Slot end is a nominal 20-minute duration (not recorded in the synthetic data).",
            "participant": [{"actor": pref, "required": "required",
                             "status": "declined" if a.status == "cancelled_patient" else "accepted"}],
        })
    for e in session.execute(select(m.Encounter).where(m.Encounter.patient_id == p.id)).scalars():
        add("Encounter", e.id, {
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB",
                      "display": "ambulatory"},
            "type": [{"text": e.encounter_type.replace("_", " ")}],
            "subject": pref, "period": {"start": _ts(e.start_at), "end": _ts(e.end_at)},
            "appointment": [{"reference": _full_url("Appointment", e.appointment_id)}]
            if e.appointment_id else None,
        })
    for o in session.execute(select(m.Observation).where(m.Observation.patient_id == p.id)).scalars():
        category = "laboratory" if o.code in LAB_CODES else "vital-signs"
        add("Observation", o.id, {
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                      "code": category}]}],
            "code": {"coding": [{"system": o.code_system, "code": o.code, "display": o.display}],
                     "text": o.display},
            "subject": pref,
            "encounter": {"reference": _full_url("Encounter", o.encounter_id)} if o.encounter_id else None,
            "effectiveDateTime": _ts(o.effective_at), "issued": _ts(o.recorded_at),
            "valueQuantity": {"value": o.value_num, "unit": o.unit, "system": UCUM, "code": o.unit},
        })
    for x in session.execute(select(m.MedicationStatement)
                             .where(m.MedicationStatement.patient_id == p.id)).scalars():
        add("MedicationStatement", x.id, {
            "status": "stopped" if x.end_date else "active",
            "medicationCodeableConcept": {"coding": [{"system": x.code_system, "code": x.code,
                                                      "display": x.display}], "text": x.display},
            "subject": pref,
            "effectivePeriod": {"start": _d(x.start_date), "end": _d(x.end_date)},
            "dateAsserted": _ts(x.recorded_at),
        })
    current = effective_scores(session, [e.id for e in session.execute(
        select(m.FollowupEpisode).where(m.FollowupEpisode.patient_id == p.id)).scalars()])
    if current:
        runs = {r.id: r for r in session.execute(select(m.ScoringRun).where(
            m.ScoringRun.id.in_({x.scoring_run_id for x in current.values()}))).scalars()}
        rows = [(r, session.get(m.FollowupEpisode, r.episode_id), session.get(m.EpisodeOutcome, r.episode_id))
                for r in current.values()]
        for r, ep, out in rows:
            run = runs[r.scoring_run_id]
            ctx = score_context(ep, out, r, as_of)
            add("RiskAssessment", f"rs_{r.id}", {
                "identifier": [{"system": "urn:cura-ai:risk-score", "value": str(r.id)}],
                "status": "final",
                "method": {"coding": [{"system": "urn:cura-ai:model", "code": r.model_version,
                                       "display": f"{run.model_type} ({run.model_version})"}],
                           "text": f"Cura AI follow-up risk model {r.model_version} ({run.model_type}), "
                                   f"feature spec {run.feature_spec_version}"},
                "code": {"text": "Risk that the chronic-care follow-up is still incomplete 30 days after "
                                 "its due date"},
                "subject": pref,
                "occurrenceDateTime": _ts(r.prediction_at),
                "prediction": [{
                    "outcome": {"text": "Follow-up not completed within 30 days of the due date"},
                    "probabilityDecimal": round(r.probability, 6),
                    "qualitativeRisk": {"text": f"{r.risk_tier} priority"},
                    "whenPeriod": {"start": _ts(r.prediction_at),
                                   "end": _ts(datetime.combine(ep.window_end_date, datetime.max.time()
                                                               .replace(microsecond=0), tzinfo=UTC))},
                }],
                "note": [{"text": (f"SIMULATED PREDICTION on synthetic data; not a clinical assessment. "
                                   f"Scoring run {run.id}, scored {_ts(r.scored_at)}, data fingerprint "
                                   f"{run.data_fingerprint[:16]}, bundle metadata sha256 "
                                   f"{run.bundle_metadata_sha256[:16]}. Score status at {as_of}: "
                                   f"{ctx.score_status}; follow-up status: {ctx.follow_up_status}.")}],
            })
    for o in session.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.patient_id == p.id)).scalars():
        mode = PARTICIPATION_MODE.get(o.channel)
        not_done = o.outcome in NOT_DONE
        add("Communication", o.id, {
            "identifier": [{"system": "urn:cura-ai:outreach", "value": o.id}],
            "status": "not-done" if not_done else "completed",
            "statusReason": {"text": o.outcome.replace("_", " ")} if not_done else None,
            "category": [{"text": f"follow-up outreach ({o.purpose.replace('_', ' ')}; "
                                  f"origin {o.origin})"}],
            "medium": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationMode",
                                    "code": mode[0], "display": mode[1]}]} if mode else {"text": o.channel}],
            "subject": pref,
            "sent": _ts(o.attempted_at), "received": None,
            "payload": [{"contentString": _payload_text(o)}],
        })
    now = datetime.now(UTC)
    return {
        "resourceType": "Bundle",
        "id": uuid.uuid4().hex,
        "meta": {"lastUpdated": _ts(now), "tag": [TAG]},
        "identifier": {"system": "urn:cura-ai:export", "value": f"{fhir_id(p.id)}-{now:%Y%m%dT%H%M%SZ}"},
        "type": "collection",
        "timestamp": _ts(now),
        "entry": entries,
    }


def structural_problems(bundle: dict[str, Any]) -> list[str]:
    """R4 structural rules relied upon by this export (not a substitute for the HL7 validator)."""
    problems: list[str] = []
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "collection":
        problems.append("not a collection Bundle")
    full_urls = set()
    for i, e in enumerate(bundle.get("entry", [])):
        r = e.get("resource", {})
        fu = e.get("fullUrl", "")
        if not re.match(r"^urn:uuid:[0-9a-f-]{36}$", fu):
            problems.append(f"entry {i}: fullUrl not a urn:uuid")
        if fu in full_urls:
            problems.append(f"entry {i}: duplicate fullUrl")
        full_urls.add(fu)
        if not FHIR_ID.match(r.get("id", "")):
            problems.append(f"entry {i}: invalid id {r.get('id')!r}")
        rt = r.get("resourceType")
        if rt == "Observation":
            q = r.get("valueQuantity", {})
            if q.get("system") != UCUM or not q.get("code"):
                problems.append(f"{r['id']}: quantity without UCUM code")
        if rt == "Appointment":
            if r["status"] in ("booked", "fulfilled", "noshow", "arrived", "checked-in") and \
                    not (r.get("start") and r.get("end")):
                problems.append(f"{r['id']}: app-3 start/end required")
            if bool(r.get("start")) != bool(r.get("end")):
                problems.append(f"{r['id']}: app-2 start and end must both be present")
            if not r.get("participant"):
                problems.append(f"{r['id']}: participant required")
        if rt == "RiskAssessment":
            for pred in r.get("prediction", []):
                pd_ = pred.get("probabilityDecimal")
                if pd_ is not None and not 0 <= pd_ <= 1:
                    problems.append(f"{r['id']}: ras-2 probability out of range")
    refs = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "reference" in node and isinstance(node["reference"], str):
                refs.append(node["reference"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(bundle.get("entry", []))
    for ref in refs:
        if ref not in full_urls:
            problems.append(f"unresolved reference {ref}")
    return problems


def r4b_parse_errors(bundle: dict[str, Any]) -> list[str]:
    """Parse every resource with the `fhir.resources` R4B models (structure, types, cardinality)."""
    from fhir.resources.R4B import get_fhir_model_class

    errors = []
    for e in bundle.get("entry", []):
        r = e["resource"]
        try:
            get_fhir_model_class(r["resourceType"]).model_validate(r)
        except Exception as exc:  # noqa: BLE001 - collect every failure
            errors.append(f"{r['resourceType']}/{r.get('id')}: {str(exc).splitlines()[0]}")
    try:
        get_fhir_model_class("Bundle").model_validate(bundle)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Bundle: {str(exc).splitlines()[0]}")
    return errors
