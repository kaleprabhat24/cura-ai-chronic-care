"""Paginated, sortable and filterable follow-up worklist (persisted episodes + latest scoring run)."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func, or_, select

from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import (
    CONDITION_LABELS,
    PageParams,
    SessionDep,
    latest_load,
    latest_run,
    patient_summary,
    prediction_summary,
    recorded_outcome,
)
from cura_api.api.core import episode_base, open_episode_filter
from cura_api.auth import CLINICAL, Principal
from cura_api.errors import ERROR_RESPONSES, ApiError

router = APIRouter(responses=ERROR_RESPONSES, tags=["worklist"])

SortKey = Literal["priority", "probability", "due_date", "last_contact", "patient_name"]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.get("/worklist", response_model=s.WorklistPage, summary="Follow-up worklist")
def worklist(
    session: SessionDep,
    _: Annotated[Principal, Depends(CLINICAL)],
    paging: Annotated[PageParams, Depends()],
    sort: Annotated[SortKey, Query(description="priority = model rank in the latest scoring "
                                               "run")] = "priority",
    order: Annotated[Literal["asc", "desc"] | None, Query(description="default depends on sort")] = None,
    q: Annotated[str | None, Query(min_length=1, max_length=80,
                                   description="search name, MRN or patient id")] = None,
    tier: Annotated[list[s.Tier] | None, Query(description="repeatable")] = None,
    condition: Annotated[list[s.Programme] | None, Query(description="programme: t2d | htn")] = None,
    due_from: date | None = None,
    due_to: date | None = None,
    overdue: bool | None = None,
    status: Annotated[Literal["open", "all"], Query(description="open = awaiting completion at the "
                                                                "data cutoff")] = "open",
) -> s.WorklistPage:
    if due_from and due_to and due_from > due_to:
        raise ApiError(422, "validation_error", "due_from must not be after due_to",
                       {"due_from": str(due_from), "due_to": str(due_to)})
    as_of = latest_load(session).data_cutoff
    run = latest_run(session)
    run_id = run.id if run else -1
    last_contact = (select(func.max(m.OutreachAttempt.attempted_at))
                    .where(m.OutreachAttempt.episode_id == m.FollowupEpisode.id)
                    .correlate(m.FollowupEpisode).scalar_subquery())
    n_outreach = (select(func.count(m.OutreachAttempt.id))
                  .where(m.OutreachAttempt.episode_id == m.FollowupEpisode.id)
                  .correlate(m.FollowupEpisode).scalar_subquery())
    query = (episode_base(m.FollowupEpisode, m.Patient, m.EpisodeOutcome, m.RiskScore,
                          last_contact.label("last_contact"), n_outreach.label("n_outreach"))
             .outerjoin(m.RiskScore, and_(m.RiskScore.episode_id == m.FollowupEpisode.id,
                                          m.RiskScore.scoring_run_id == run_id))
             .where(m.FollowupEpisode.eligible.is_(True)))
    if status == "open":
        query = query.where(open_episode_filter())
    if q:
        pattern = f"%{_escape_like(q.strip())}%"
        full_name = m.Patient.given_name + " " + m.Patient.family_name
        query = query.where(or_(*(col.ilike(pattern, escape="\\") for col in (
            m.Patient.given_name, m.Patient.family_name, full_name, m.Patient.synthetic_mrn,
            m.Patient.id))))
    if tier:
        query = query.where(m.RiskScore.risk_tier.in_(tier))
    if condition:
        query = query.where(m.FollowupEpisode.programme.in_(condition))
    if due_from:
        query = query.where(m.FollowupEpisode.due_date >= due_from)
    if due_to:
        query = query.where(m.FollowupEpisode.due_date <= due_to)
    if overdue is not None:
        query = query.where(m.FollowupEpisode.due_date < as_of if overdue
                            else m.FollowupEpisode.due_date >= as_of)

    default_desc = {"priority": False, "probability": True, "due_date": False, "last_contact": True,
                    "patient_name": False}
    desc = default_desc[sort] if order is None else order == "desc"
    keys = {"priority": [m.RiskScore.rank], "probability": [m.RiskScore.probability],
            "due_date": [m.FollowupEpisode.due_date], "last_contact": [last_contact],
            "patient_name": [m.Patient.family_name, m.Patient.given_name]}[sort]
    ordering = [(k.desc() if desc else k.asc()).nulls_last() for k in keys] + [m.FollowupEpisode.id.asc()]

    total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()
    rows = session.execute(query.order_by(*ordering).offset(paging.offset).limit(paging.page_size)).all()

    ids = [r[0].id for r in rows]
    last: dict[str, m.OutreachAttempt] = {}
    if ids:
        for o in session.execute(select(m.OutreachAttempt).where(m.OutreachAttempt.episode_id.in_(ids))
                                 .order_by(m.OutreachAttempt.attempted_at)).scalars():
            last[o.episode_id] = o
    items = []
    for ep, pt, out, rs, _lc, n in rows:
        lo = last.get(ep.id)
        items.append(s.WorklistItem(
            episode_id=ep.id, programme=ep.programme, condition=CONDITION_LABELS[ep.programme],
            due_date=ep.due_date, days_until_due=(ep.due_date - as_of).days,
            prediction_at=ep.prediction_at, patient=patient_summary(pt, as_of),
            prediction=prediction_summary(rs),
            last_outreach=None if lo is None else s.OutreachSummary(
                id=lo.id, attempted_at=lo.attempted_at, channel=lo.channel, outcome=lo.outcome,
                origin=lo.origin),
            outreach_attempts=n or 0, recorded_outcome=recorded_outcome(ep, out, as_of)))
    return s.WorklistPage(items=items, as_of=as_of, scoring_run_id=run.id if run else None,
                          **paging.meta(total))
