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


def prediction_summary(rs: m.RiskScore | None) -> s.PredictionSummary | None:
    if rs is None:
        return None
    factors = (rs.explanation or {}).get("factors") or []
    return s.PredictionSummary(risk_score_id=rs.id, scoring_run_id=rs.scoring_run_id,
                               probability=rs.probability, risk_tier=rs.risk_tier, rank=rs.rank,
                               model_version=rs.model_version, prediction_at=rs.prediction_at,
                               scored_at=rs.scored_at,
                               top_factor=s.Factor(**factors[0]) if factors else None)


def run_out(run: m.ScoringRun | None) -> s.ScoringRunOut | None:
    return None if run is None else s.ScoringRunOut.model_validate(run)
