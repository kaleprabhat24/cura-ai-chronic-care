"""Patient onboarding endpoints: patient list, registration, history entries and score status."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy import func, or_, select

from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import (
    APP_PATIENT_PREFIX,
    PageParams,
    SessionDep,
    is_app_patient,
    latest_load,
    patient_summary,
    reference_date,
    today,
)
from cura_api.auth import CLINICAL, Principal
from cura_api.errors import ERROR_RESPONSES, ErrorResponse
from cura_api.services import onboarding
from cura_ml.synth import catalog as cat

router = APIRouter(responses=ERROR_RESPONSES, tags=["patients"])
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Not found"}}
CONFLICT = {409: {"model": ErrorResponse, "description": "Possible duplicate or conflicting record"}}
CODE_TO_PROGRAMME = {c["code"]: k for k, c in cat.CONDITIONS.items()}
APP_LIKE = APP_PATIENT_PREFIX.replace("_", "\\_") + "%"


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _now(request: Request):
    return request.app.state.clock()


@router.get("/patients", response_model=s.Page[s.PatientListItem], summary="Patient list and search")
def list_patients(
    session: SessionDep,
    request: Request,
    _: Annotated[Principal, Depends(CLINICAL)],
    paging: Annotated[PageParams, Depends()],
    q: Annotated[
        str | None, Query(min_length=1, max_length=80, description="name, MRN or patient id")
    ] = None,
    origin: Annotated[
        Literal["all", "registered", "synthetic"], Query(description="registered = added in Cura AI")
    ] = "all",
    sort: Literal["name", "recently_registered"] = "name",
) -> s.Page[s.PatientListItem]:
    as_of = latest_load(session).data_cutoff
    now_day = today(request)
    base = select(m.Patient)
    if q:
        pattern = f"%{_escape_like(q.strip())}%"
        full_name = m.Patient.given_name + " " + m.Patient.family_name
        base = base.where(
            or_(
                *(
                    col.ilike(pattern, escape="\\")
                    for col in (
                        m.Patient.given_name,
                        m.Patient.family_name,
                        full_name,
                        m.Patient.synthetic_mrn,
                        m.Patient.id,
                    )
                )
            )
        )
    if origin == "registered":
        base = base.where(m.Patient.id.like(APP_LIKE, escape="\\"))
    elif origin == "synthetic":
        base = base.where(~m.Patient.id.like(APP_LIKE, escape="\\"))
    order = (
        [m.Patient.registered_on.desc(), m.Patient.id.desc()]
        if sort == "recently_registered"
        else [m.Patient.family_name, m.Patient.given_name, m.Patient.id]
    )
    total = session.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    patients = (
        session.execute(base.order_by(*order).offset(paging.offset).limit(paging.page_size)).scalars().all()
    )
    ids = [p.id for p in patients]
    progs: dict[str, set[str]] = {i: set() for i in ids}
    for pid, code in session.execute(
        select(m.Condition.patient_id, m.Condition.code).where(m.Condition.patient_id.in_(ids))
    ).all():
        if code in CODE_TO_PROGRAMME:
            progs[pid].add(CODE_TO_PROGRAMME[code])
    next_due: dict[str, object] = {}
    for pid, due in session.execute(
        select(m.FollowupEpisode.patient_id, func.max(m.FollowupEpisode.due_date))
        .where(m.FollowupEpisode.patient_id.in_(ids))
        .group_by(m.FollowupEpisode.patient_id)
    ).all():
        next_due[pid] = due
    items = [
        s.PatientListItem(
            patient=patient_summary(p, reference_date(p.id, as_of, now_day)),
            registered_in_app=is_app_patient(p.id),
            programmes=sorted(progs[p.id]),
            registered_on=p.registered_on,
            next_due_date=next_due.get(p.id),
        )
        for p in patients
    ]
    return s.Page[s.PatientListItem](items=items, **paging.meta(total))


@router.post(
    "/patients",
    response_model=s.OnboardingResult,
    status_code=status.HTTP_201_CREATED,
    summary="Register a fictional patient with conditions and longitudinal history",
    responses={
        **CONFLICT,
        200: {"model": s.OnboardingResult, "description": "Submission already processed (safe replay)"},
    },
)
def register_patient(
    body: s.PatientRegistration,
    request: Request,
    response: Response,
    session: SessionDep,
    principal: Annotated[Principal, Depends(CLINICAL)],
) -> s.OnboardingResult:
    result = onboarding.register_patient(
        session, body, principal, _now(request), request.app.state.settings.model_bundle_dir
    )
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return result


@router.post(
    "/patients/{patient_id}/history",
    response_model=s.OnboardingResult,
    status_code=status.HTTP_201_CREATED,
    summary="Add clinical-history entries to a patient",
    responses={
        **NOT_FOUND,
        200: {"model": s.OnboardingResult, "description": "Submission already processed (safe replay)"},
    },
)
def add_history(
    patient_id: str,
    body: s.HistoryAddition,
    request: Request,
    response: Response,
    session: SessionDep,
    principal: Annotated[Principal, Depends(CLINICAL)],
) -> s.OnboardingResult:
    result = onboarding.add_history(
        session, patient_id, body, principal, _now(request), request.app.state.settings.model_bundle_dir
    )
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return result


@router.get(
    "/patients/{patient_id}/score-status",
    response_model=s.ScoreStatusOut,
    responses=NOT_FOUND,
    summary="Why a registered patient does or does not have a risk score (read-only)",
)
def get_score_status(
    patient_id: str, request: Request, session: SessionDep, _: Annotated[Principal, Depends(CLINICAL)]
) -> s.ScoreStatusOut:
    return onboarding.score_status(
        session, patient_id, _now(request), request.app.state.settings.model_bundle_dir
    )


@router.post(
    "/patients/{patient_id}/score",
    response_model=s.ScoreStatusOut,
    responses={**NOT_FOUND, **CONFLICT},
    summary="Score a registered patient's follow-ups whose prediction time has arrived",
)
def post_score(
    patient_id: str, request: Request, session: SessionDep, principal: Annotated[Principal, Depends(CLINICAL)]
) -> s.ScoreStatusOut:
    return onboarding.generate_scores(
        session, patient_id, principal, _now(request), request.app.state.settings.model_bundle_dir
    )
