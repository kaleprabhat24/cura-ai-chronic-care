"""Patient profile, longitudinal timeline, episodes, risk scores and outreach."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time
from typing import Annotated, Literal, get_args

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import (
    CONDITION_LABELS,
    SessionDep,
    age_at,
    latest_load,
    latest_run,
    prediction_summary,
    recorded_outcome,
    run_out,
)
from cura_api.auth import CLINICAL, Principal
from cura_api.db import utcnow
from cura_api.errors import ERROR_RESPONSES, ApiError, ErrorResponse
from cura_api.services import audit
from cura_ml.features import spec as S

router = APIRouter(responses=ERROR_RESPONSES)
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Not found"}}
EVENT_TYPES = get_args(s.EventType)


def _patient(session: Session, patient_id: str) -> m.Patient:
    p = session.get(m.Patient, patient_id)
    if p is None:
        raise ApiError(404, "not_found", f"Patient '{patient_id}' not found.")
    return p


def _episode(session: Session, episode_id: str) -> tuple[m.FollowupEpisode, m.EpisodeOutcome]:
    ep = session.get(m.FollowupEpisode, episode_id)
    if ep is None:
        raise ApiError(404, "not_found", f"Episode '{episode_id}' not found.")
    return ep, session.get(m.EpisodeOutcome, episode_id)


def _latest_scores(session: Session, episode_ids: list[str]) -> dict[str, m.RiskScore]:
    run = latest_run(session)
    if run is None or not episode_ids:
        return {}
    rows = session.execute(select(m.RiskScore).where(m.RiskScore.scoring_run_id == run.id,
                                                      m.RiskScore.episode_id.in_(episode_ids))).scalars()
    return {r.episode_id: r for r in rows}


def _episode_out(ep: m.FollowupEpisode, out: m.EpisodeOutcome, as_of: date,
                 score: m.RiskScore | None) -> s.EpisodeOut:
    return s.EpisodeOut(id=ep.id, patient_id=ep.patient_id, programme=ep.programme,
                        condition=CONDITION_LABELS[ep.programme], index_event_at=ep.index_event_at,
                        due_date=ep.due_date, prediction_at=ep.prediction_at,
                        window_end_date=ep.window_end_date, eligible=ep.eligible,
                        ineligible_reason=ep.ineligible_reason, is_cold_start=ep.is_cold_start,
                        recorded_outcome=recorded_outcome(ep, out, as_of),
                        latest_prediction=prediction_summary(score))


# ------------------------------------------------------------------------------------------ patients
@router.get("/patients/{patient_id}", response_model=s.PatientDetail, tags=["patients"],
            summary="Patient profile (Patient 360)", responses=NOT_FOUND)
def patient_detail(patient_id: str, session: SessionDep,
                   _: Annotated[Principal, Depends(CLINICAL)]) -> s.PatientDetail:
    as_of = latest_load(session).data_cutoff
    p = _patient(session, patient_id)
    conditions = session.execute(select(m.Condition).where(m.Condition.patient_id == p.id)
                                 .order_by(m.Condition.recorded_at)).scalars().all()
    eps = session.execute(select(m.FollowupEpisode, m.EpisodeOutcome)
                          .join(m.EpisodeOutcome, m.EpisodeOutcome.episode_id == m.FollowupEpisode.id)
                          .where(m.FollowupEpisode.patient_id == p.id)
                          .order_by(m.FollowupEpisode.index_event_at.desc())).all()
    scores = _latest_scores(session, [e.id for e, _ in eps])
    meds = session.execute(select(m.MedicationStatement).where(
        m.MedicationStatement.patient_id == p.id, m.MedicationStatement.end_date.is_(None))
        .order_by(m.MedicationStatement.start_date)).scalars().all()
    obs = session.execute(select(m.Observation).where(m.Observation.patient_id == p.id)
                          .order_by(m.Observation.effective_at)).scalars().all()
    latest_obs: dict[str, m.Observation] = {}
    for o in obs:
        latest_obs[o.code] = o
    counts = {
        "appointments": len(session.execute(select(m.Appointment.id).where(
            m.Appointment.patient_id == p.id)).all()),
        "encounters": len(session.execute(select(m.Encounter.id).where(
            m.Encounter.patient_id == p.id)).all()),
        "observations": len(obs),
        "outreach_attempts": len(session.execute(select(m.OutreachAttempt.id).where(
            m.OutreachAttempt.patient_id == p.id)).all()),
        "episodes": len(eps),
    }
    return s.PatientDetail(
        as_of=as_of,
        patient=s.PatientOut(id=p.id, synthetic_mrn=p.synthetic_mrn, given_name=p.given_name,
                             family_name=p.family_name, birth_date=p.birth_date,
                             age_years=age_at(p.birth_date, as_of), sex=p.sex,
                             deprivation_quintile=p.deprivation_quintile,
                             preferred_language=p.preferred_language, registered_on=p.registered_on,
                             death_date=p.death_date, transfer_out_date=p.transfer_out_date,
                             is_synthetic=p.is_synthetic),
        conditions=[s.ConditionOut(id=c.id, code_system=c.code_system, code=c.code, display=c.display,
                                   onset_date=c.onset_date, recorded_at=c.recorded_at,
                                   clinical_status=c.clinical_status) for c in conditions],
        episodes=[_episode_out(e, o, as_of, scores.get(e.id)) for e, o in eps],
        active_medications=[s.MedicationOut(id=x.id, code=x.code, display=x.display,
                                            drug_class=x.drug_class, start_date=x.start_date,
                                            end_date=x.end_date, recorded_at=x.recorded_at) for x in meds],
        latest_observations=[s.ObservationOut(id=o.id, code=o.code, display=o.display, value=o.value_num,
                                              unit=o.unit, effective_at=o.effective_at,
                                              recorded_at=o.recorded_at) for o in latest_obs.values()],
        counts=counts)


@router.get("/patients/{patient_id}/timeline", response_model=s.Timeline, tags=["patients"],
            summary="Unified longitudinal timeline", responses=NOT_FOUND)
def timeline(
    patient_id: str, session: SessionDep, _: Annotated[Principal, Depends(CLINICAL)],
    types: Annotated[list[s.EventType] | None, Query(description="repeatable filter")] = None,
    date_from: Annotated[date | None, Query(alias="from")] = None,
    date_to: Annotated[date | None, Query(alias="to")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 300,
    order: Literal["asc", "desc"] = "desc",
) -> s.Timeline:
    p = _patient(session, patient_id)
    wanted = set(types or EVENT_TYPES)
    ev: list[s.TimelineEvent] = []
    clin, outreach, pred = m.PROVENANCE_CLINICAL, m.PROVENANCE_OUTREACH, m.PROVENANCE_PREDICTION
    if "appointment" in wanted:
        for a in session.execute(select(m.Appointment).where(m.Appointment.patient_id == p.id)).scalars():
            ev.append(s.TimelineEvent(
                id=a.id, type="appointment", occurred_at=a.scheduled_for, recorded_at=a.status_recorded_at,
                title=f"{a.appointment_type.replace('_', ' ')} appointment — {a.status.replace('_', ' ')}",
                detail={"status": a.status, "booked_at": a.booked_at.isoformat(),
                        "booking_source": a.booking_source, "appointment_type": a.appointment_type},
                provenance=clin))
    if "encounter" in wanted:
        for e in session.execute(select(m.Encounter).where(m.Encounter.patient_id == p.id)).scalars():
            ev.append(s.TimelineEvent(id=e.id, type="encounter", occurred_at=e.start_at,
                                      recorded_at=e.recorded_at,
                                      title=f"{e.encounter_type.replace('_', ' ')} visit",
                                      detail={"encounter_type": e.encounter_type,
                                              "end_at": e.end_at.isoformat(),
                                              "appointment_id": e.appointment_id}, provenance=clin))
    if "observation" in wanted:
        for o in session.execute(select(m.Observation).where(m.Observation.patient_id == p.id)).scalars():
            ev.append(s.TimelineEvent(id=o.id, type="observation", occurred_at=o.effective_at,
                                      recorded_at=o.recorded_at,
                                      title=f"{o.display}: {o.value_num:g} {o.unit}",
                                      detail={"code": o.code, "code_system": o.code_system,
                                              "value": o.value_num, "unit": o.unit,
                                              "encounter_id": o.encounter_id}, provenance=clin))
    if wanted & {"medication_start", "medication_stop"}:
        for x in session.execute(select(m.MedicationStatement)
                                 .where(m.MedicationStatement.patient_id == p.id)).scalars():
            if "medication_start" in wanted:
                ev.append(s.TimelineEvent(
                    id=f"{x.id}:start", type="medication_start",
                    occurred_at=datetime.combine(x.start_date, time.min, tzinfo=UTC),
                    recorded_at=x.recorded_at, title=f"Started {x.display}",
                    detail={"code": x.code, "drug_class": x.drug_class}, provenance=clin))
            if "medication_stop" in wanted and x.end_date is not None:
                ev.append(s.TimelineEvent(
                    id=f"{x.id}:stop", type="medication_stop",
                    occurred_at=datetime.combine(x.end_date, time.min, tzinfo=UTC),
                    recorded_at=x.end_recorded_at, title=f"Stopped {x.display}",
                    detail={"code": x.code, "drug_class": x.drug_class}, provenance=clin))
    if "outreach" in wanted:
        for o in session.execute(select(m.OutreachAttempt)
                                 .where(m.OutreachAttempt.patient_id == p.id)).scalars():
            ev.append(s.TimelineEvent(
                id=o.id, type="outreach", occurred_at=o.attempted_at, recorded_at=o.recorded_at,
                title=f"Outreach ({o.channel}) — {o.outcome.replace('_', ' ')}",
                detail={"origin": o.origin, "purpose": o.purpose, "note": o.note,
                        "next_action": o.next_action,
                        "next_action_due": o.next_action_due.isoformat() if o.next_action_due else None,
                        "recorded_by": o.recorded_by}, provenance=outreach, episode_id=o.episode_id))
    if wanted & {"risk_score", "followup_due"}:
        eps = session.execute(select(m.FollowupEpisode).where(m.FollowupEpisode.patient_id == p.id,
                                                               m.FollowupEpisode.eligible.is_(True))).scalars().all()
        if "followup_due" in wanted:
            for e in eps:
                ev.append(s.TimelineEvent(
                    id=f"{e.id}:due", type="followup_due",
                    occurred_at=datetime.combine(e.due_date, time.min, tzinfo=UTC), recorded_at=None,
                    title=f"{CONDITION_LABELS[e.programme]} review due",
                    detail={"prediction_at": e.prediction_at.isoformat(),
                            "window_end_date": e.window_end_date.isoformat()},
                    provenance=clin, episode_id=e.id))
        if "risk_score" in wanted:
            for r in session.execute(select(m.RiskScore).where(m.RiskScore.patient_id == p.id)).scalars():
                ev.append(s.TimelineEvent(
                    id=f"rs_{r.id}", type="risk_score", occurred_at=r.prediction_at, recorded_at=r.scored_at,
                    title=f"Simulated risk score: {r.risk_tier} ({r.probability:.0%})",
                    detail={"risk_score_id": r.id, "scoring_run_id": r.scoring_run_id,
                            "model_version": r.model_version, "probability": r.probability},
                    provenance=pred, episode_id=r.episode_id))
    if date_from:
        ev = [e for e in ev if e.occurred_at.date() >= date_from]
    if date_to:
        ev = [e for e in ev if e.occurred_at.date() <= date_to]
    ev.sort(key=lambda e: (e.occurred_at, e.id), reverse=order == "desc")
    return s.Timeline(patient_id=p.id, total=len(ev), returned=min(limit, len(ev)), items=ev[:limit])


@router.get("/patients/{patient_id}/outreach-attempts", response_model=list[s.OutreachOut],
            tags=["outreach"], summary="Outreach history for a patient", responses=NOT_FOUND)
def patient_outreach(patient_id: str, session: SessionDep,
                     _: Annotated[Principal, Depends(CLINICAL)]) -> list[s.OutreachOut]:
    _patient(session, patient_id)
    rows = session.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.patient_id == patient_id)
                           .order_by(m.OutreachAttempt.attempted_at.desc())).scalars()
    return [s.OutreachOut.model_validate(r) for r in rows]


# ------------------------------------------------------------------------------------------ episodes
@router.get("/episodes/{episode_id}", response_model=s.EpisodeOut, tags=["episodes"],
            summary="Follow-up episode with recorded outcome and latest prediction", responses=NOT_FOUND)
def episode_detail(episode_id: str, session: SessionDep,
                   _: Annotated[Principal, Depends(CLINICAL)]) -> s.EpisodeOut:
    as_of = latest_load(session).data_cutoff
    ep, out = _episode(session, episode_id)
    return _episode_out(ep, out, as_of, _latest_scores(session, [ep.id]).get(ep.id))


@router.get("/episodes/{episode_id}/risk-scores", response_model=list[s.RiskScoreHistoryItem],
            tags=["risk scores"], summary="All scoring snapshots for an episode", responses=NOT_FOUND)
def episode_scores(episode_id: str, session: SessionDep,
                   _: Annotated[Principal, Depends(CLINICAL)]) -> list[s.RiskScoreHistoryItem]:
    _episode(session, episode_id)
    rows = session.execute(select(m.RiskScore).where(m.RiskScore.episode_id == episode_id)
                           .order_by(m.RiskScore.scoring_run_id.desc())).scalars()
    return [s.RiskScoreHistoryItem(id=r.id, scoring_run_id=r.scoring_run_id, model_version=r.model_version,
                                   probability=r.probability, risk_tier=r.risk_tier, rank=r.rank,
                                   prediction_at=r.prediction_at, scored_at=r.scored_at) for r in rows]


@router.get("/risk-scores/{risk_score_id}", response_model=s.RiskScoreDetail, tags=["risk scores"],
            summary="Risk score with the model's feature explanation and provenance", responses=NOT_FOUND)
def risk_score_detail(risk_score_id: int, session: SessionDep,
                      _: Annotated[Principal, Depends(CLINICAL)]) -> s.RiskScoreDetail:
    r = session.get(m.RiskScore, risk_score_id)
    if r is None:
        raise ApiError(404, "not_found", f"Risk score {risk_score_id} not found.")
    run = session.get(m.ScoringRun, r.scoring_run_id)
    feats = []
    for col, value in r.feature_snapshot.items():
        base = S.FEATURE_BY_NAME[S.base_feature(col)]
        feats.append(s.FeatureValue(column=col, label=S.column_label(col),
                                    unit="yes/no" if col.endswith("__missing") else base.unit, value=value))
    e = r.explanation
    return s.RiskScoreDetail(
        id=r.id, scoring_run_id=r.scoring_run_id, episode_id=r.episode_id, patient_id=r.patient_id,
        model_version=r.model_version, model_type=run.model_type, probability=r.probability,
        risk_tier=r.risk_tier, rank=r.rank, model_log_odds=r.model_log_odds, prediction_at=r.prediction_at,
        scored_at=r.scored_at,
        explanation=s.Explanation(method=e["method"], base_log_odds=e["base_log_odds"],
                                  model_log_odds=e["model_log_odds"],
                                  factors=[s.Factor(**f) for f in e["factors"]], units=e["units"],
                                  note=e["note"]),
        features=feats, run=run_out(run),
        interpretation=("Simulated prediction from a model trained on synthetic data: the calibrated "
                        "probability that this follow-up is still incomplete 30 days after its due date. "
                        "Feature values are those recorded before the prediction time. It is not a "
                        "diagnosis or a clinical assessment."))


@router.get("/episodes/{episode_id}/outreach-attempts", response_model=list[s.OutreachOut],
            tags=["outreach"], summary="Outreach history for an episode", responses=NOT_FOUND)
def episode_outreach(episode_id: str, session: SessionDep,
                     _: Annotated[Principal, Depends(CLINICAL)]) -> list[s.OutreachOut]:
    _episode(session, episode_id)
    rows = session.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.episode_id == episode_id)
                           .order_by(m.OutreachAttempt.attempted_at.desc())).scalars()
    return [s.OutreachOut.model_validate(r) for r in rows]


@router.post("/episodes/{episode_id}/outreach-attempts", response_model=s.OutreachOut,
             status_code=status.HTTP_201_CREATED, tags=["outreach"],
             summary="Record an outreach attempt and its outcome",
             responses={**NOT_FOUND, 409: {"description": "Episode not eligible for outreach"}})
def create_outreach(episode_id: str, body: s.OutreachCreate, session: SessionDep,
                    principal: Annotated[Principal, Depends(CLINICAL)]) -> s.OutreachOut:
    ep, _ = _episode(session, episode_id)
    if not ep.eligible:
        raise ApiError(409, "conflict", "Outreach can only be recorded for eligible follow-up episodes.",
                       {"ineligible_reason": ep.ineligible_reason})
    now = utcnow()
    attempt = m.OutreachAttempt(
        id=f"oa_app_{uuid.uuid4().hex}", patient_id=ep.patient_id, episode_id=ep.id,
        programme=ep.programme, origin="app", purpose="coordinator_outreach", channel=body.channel,
        outcome=body.outcome, note=body.note, next_action=body.next_action,
        next_action_due=body.next_action_due, attempted_at=body.attempted_at or now, recorded_at=now,
        recorded_by=principal.user_id, row_hash=None)
    try:
        session.add(attempt)
        session.flush()
        audit.record(session, principal.user_id, principal.role, "outreach.create", "outreach_attempt",
                     attempt.id, {"episode_id": ep.id, "patient_id": ep.patient_id,
                                  "channel": body.channel, "outcome": body.outcome,
                                  "next_action": body.next_action})
        session.commit()
    except Exception:
        session.rollback()
        raise
    return s.OutreachOut.model_validate(attempt)
