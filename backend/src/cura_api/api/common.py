"""Shared API dependencies and helpers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy import func, select
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


BATCH_SCOPES = ("eligible", "open")   # cohort-wide `cura-api score` runs
PATIENT_SCOPE = "patient"              # per-patient runs for patients registered in the app
APP_PATIENT_PREFIX = "pt_app_"


def latest_run(session: Session) -> m.ScoringRun | None:
    """The latest cohort-wide (batch) scoring run. Per-patient runs never replace it."""
    return session.execute(select(m.ScoringRun).where(m.ScoringRun.status == "succeeded",
                                                      m.ScoringRun.scope.in_(BATCH_SCOPES))
                           .order_by(m.ScoringRun.id.desc())).scalars().first()


def is_app_patient(patient_id: str) -> bool:
    return patient_id.startswith(APP_PATIENT_PREFIX)


def today(request: Request) -> date:
    """Current UTC date from the application clock (overridable in tests)."""
    return request.app.state.clock().date()


def reference_date(patient_id: str, dataset_as_of: date, today_: date) -> date:
    """The date against which an episode's status is judged. The synthetic cohort lives at the dataset
    cutoff ("simulated today"); patients registered in the app live in real time."""
    return max(dataset_as_of, today_) if is_app_patient(patient_id) else dataset_as_of


def effective_run_id(batch_run_id: int | None):
    """SQL expression (correlated on FollowupEpisode) for the scoring run whose score is current for an
    episode: the newest per-patient run that scored it (patients registered in the app), otherwise the
    latest batch run (synthetic cohort). Older snapshots stay stored as history."""
    newest_patient_run = (
        select(func.max(m.RiskScore.scoring_run_id))
        .join(m.ScoringRun, m.ScoringRun.id == m.RiskScore.scoring_run_id)
        .where(m.RiskScore.episode_id == m.FollowupEpisode.id, m.ScoringRun.scope == PATIENT_SCOPE,
               m.ScoringRun.status == "succeeded")
        .correlate(m.FollowupEpisode).scalar_subquery())
    return func.coalesce(newest_patient_run, batch_run_id if batch_run_id is not None else -1)


def effective_scores(session: Session, episode_ids: list[str]) -> dict[str, m.RiskScore]:
    """Current score per episode (same rule as `effective_run_id`), for a handful of episodes."""
    if not episode_ids:
        return {}
    batch = latest_run(session)
    out: dict[str, m.RiskScore] = {}
    if batch is not None:
        for r in session.execute(select(m.RiskScore).where(m.RiskScore.scoring_run_id == batch.id,
                                                            m.RiskScore.episode_id.in_(episode_ids))).scalars():
            out[r.episode_id] = r
    query = (select(m.RiskScore).join(m.ScoringRun, m.ScoringRun.id == m.RiskScore.scoring_run_id)
             .where(m.RiskScore.episode_id.in_(episode_ids), m.ScoringRun.scope == PATIENT_SCOPE,
                    m.ScoringRun.status == "succeeded")
             .order_by(m.RiskScore.scoring_run_id))
    rows = session.execute(query).scalars()
    for r in rows:
        out[r.episode_id] = r
    return out


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
