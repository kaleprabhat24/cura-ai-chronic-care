"""Shared API dependencies and helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from cura_api import models as m
from cura_api import schemas as s
from cura_api.errors import ApiError
from cura_ml.features import spec as S

CONDITION_LABELS = {"t2d": "Type 2 diabetes", "htn": "Hypertension"}


def get_session(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


class PageParams:
    def __init__(self, page: Annotated[int, Query(ge=1, description="1-based page number")] = 1,
                 page_size: Annotated[int, Query(ge=1, le=100)] = 25) -> None:
        self.page, self.page_size = page, page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    def meta(self, total: int) -> dict[str, int]:
        return {"page": self.page, "page_size": self.page_size, "total": total,
                "total_pages": max(1, math.ceil(total / self.page_size)) if total else 0}


def latest_load(session: Session) -> m.DatasetLoad:
    load = session.execute(select(m.DatasetLoad).order_by(m.DatasetLoad.id.desc())).scalars().first()
    if load is None:
        raise ApiError(503, "dataset_not_loaded", "No cohort has been loaded; run `cura-api seed`.")
    return load


def latest_run(session: Session) -> m.ScoringRun | None:
    return session.execute(select(m.ScoringRun).where(m.ScoringRun.status == "succeeded")
                           .order_by(m.ScoringRun.id.desc())).scalars().first()


def age_at(birth: date, on: date) -> int:
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def patient_summary(p: m.Patient, as_of: date) -> s.PatientSummary:
    return s.PatientSummary(id=p.id, synthetic_mrn=p.synthetic_mrn,
                            display_name=f"{p.given_name} {p.family_name}",
                            age_years=age_at(p.birth_date, as_of), sex=p.sex)


def outcome_status(ep: m.FollowupEpisode, out: m.EpisodeOutcome, as_of: date) -> str:
    if not ep.eligible:
        return "not_eligible"
    if out.censored:
        return "censored"
    if out.completed_at is not None:
        return "completed_on_time" if out.completed_in_window else "completed_late"
    return "open_overdue" if ep.due_date < as_of else "open"


def recorded_outcome(ep: m.FollowupEpisode, out: m.EpisodeOutcome, as_of: date) -> s.RecordedOutcome:
    return s.RecordedOutcome(status=outcome_status(ep, out, as_of), completed_at=out.completed_at,
                             completed_in_window=out.completed_in_window, label_status=out.label_status,
                             censor_reason=out.censor_reason)


def factor(stored: dict) -> s.Factor:
    """A stored explanation factor, enriched with the feature's look-back window from the spec."""
    spec = S.FEATURE_BY_NAME.get(stored.get("feature", ""))
    return s.Factor(**stored, window=spec.window if spec else None, source=spec.source if spec else None)


def prediction_summary(rs: m.RiskScore | None) -> s.PredictionSummary | None:
    if rs is None:
        return None
    factors = (rs.explanation or {}).get("factors") or []
    return s.PredictionSummary(risk_score_id=rs.id, scoring_run_id=rs.scoring_run_id,
                               probability=rs.probability, risk_tier=rs.risk_tier, rank=rs.rank,
                               model_version=rs.model_version, prediction_at=rs.prediction_at,
                               scored_at=rs.scored_at,
                               top_factor=factor(factors[0]) if factors else None)


def follow_up_status(ep: m.FollowupEpisode, out: m.EpisodeOutcome, as_of: date) -> str:
    if not ep.eligible:
        return "not_eligible"
    if out.censored:
        return "censored"
    if out.completed_at is not None:
        return "completed"
    if as_of < ep.due_date:
        return "upcoming"
    if as_of <= ep.window_end_date:
        return "due_in_grace_window"
    return "overdue_unresolved"


def score_context(ep: m.FollowupEpisode, out: m.EpisodeOutcome, rs: m.RiskScore | None,
                  as_of: date) -> s.ScoreContext:
    """Separate predictive risk from recorded follow-up status (see schemas.ScoreContext).

    A probability is only *predictive* while the episode's outcome is still unknown at the data
    cutoff: the follow-up is open and its outcome window (due date + 30 days) has not ended. Once the
    window has ended, or the episode is completed or censored, the score is *historical*. It was
    computed at the episode's prediction time and is kept for provenance, never shown as a fresh
    prediction. An open episode past its window is an observed outcome (not completed in time) that
    needs operational follow-up.
    """
    fus = follow_up_status(ep, out, as_of)
    if rs is None:
        status = "no_score"
    elif fus in ("upcoming", "due_in_grace_window"):
        status = "predictive"
    else:
        status = "historical"
    window_ended = as_of > ep.window_end_date
    days_overdue = max(0, (as_of - ep.due_date).days)
    age = None if rs is None else (as_of - rs.prediction_at.date()).days
    pred_day = rs.prediction_at.date().isoformat() if rs else None
    texts = {
        "upcoming": f"Follow-up due {ep.due_date.isoformat()}.",
        "due_in_grace_window": (f"Due date passed {days_overdue} days ago; the completion window stays "
                                f"open until {ep.window_end_date.isoformat()}."),
        "overdue_unresolved": (f"Recorded outcome: not completed by the end of the outcome window "
                               f"({ep.window_end_date.isoformat()}). Needs operational follow-up."),
        "completed": "Recorded outcome: follow-up completed"
                     + (f" on {out.completed_at.date().isoformat()}." if out.completed_at else "."),
        "censored": "Patient left care during the outcome window (death or transfer).",
        "not_eligible": f"Not eligible for prediction ({ep.ineligible_reason}).",
    }
    summary = texts[fus]
    if status == "predictive":
        summary += f" Predictive score from {pred_day} (outcome not yet known)."
    elif status == "historical":
        summary += f" The score from {pred_day} is historical, not a current prediction."
    else:
        summary += " No score in the latest scoring run."
    limited = bool(ep.is_cold_start)
    if limited:
        summary += " Limited history (< 180 days) at prediction time."
    return s.ScoreContext(as_of=as_of, follow_up_status=fus, score_status=status,
                          outcome_window_ended=window_ended, outcome_mature=bool(out.mature),
                          days_overdue=days_overdue, score_age_days=age,
                          data_sufficiency="limited_history" if limited else "adequate", summary=summary)


def run_out(run: m.ScoringRun | None) -> s.ScoringRunOut | None:
    return None if run is None else s.ScoringRunOut.model_validate(run)
