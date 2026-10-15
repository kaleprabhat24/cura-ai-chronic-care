"""Health, identity and dashboard endpoints."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import and_, func, select, text

from cura_api import __version__
from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import SessionDep, latest_load, latest_run, run_out
from cura_api.auth import ANY_ROLE, Principal
from cura_api.errors import ERROR_RESPONSES

router = APIRouter(responses=ERROR_RESPONSES)


def open_episode_filter():
    """Eligible episodes still awaiting completion at the data cutoff."""
    return and_(m.FollowupEpisode.eligible.is_(True), m.EpisodeOutcome.completed_at.is_(None),
                m.EpisodeOutcome.censored.is_(False), m.Patient.death_date.is_(None),
                m.Patient.transfer_out_date.is_(None))


def episode_base(*cols):
    return (select(*cols).select_from(m.FollowupEpisode)
            .join(m.Patient, m.Patient.id == m.FollowupEpisode.patient_id)
            .join(m.EpisodeOutcome, m.EpisodeOutcome.episode_id == m.FollowupEpisode.id))


@router.get("/health", response_model=s.Health, tags=["system"], summary="Service health")
def health(request: Request, session: SessionDep) -> s.Health:
    settings = request.app.state.settings
    db_ok, loaded, run_id = "ok", False, None
    try:
        session.execute(text("SELECT 1"))
        loaded = session.execute(select(func.count()).select_from(m.DatasetLoad)).scalar_one() > 0
        run = latest_run(session)
        run_id = run.id if run else None
    except Exception:  # noqa: BLE001 - health must report, not raise
        db_ok = "error"
    bundle_ok = (settings.model_bundle_dir / "metadata.json").exists()
    status = "ok" if db_ok == "ok" and loaded and bundle_ok and run_id else "degraded"
    return s.Health(
    status=status,
    version=__version__,
    environment=settings.environment,
    auth_mode=settings.auth_mode,
    production_auth_implemented=(
        settings.auth_mode == "supabase"
        and request.app.state.token_verifier is not None
    ),
    database=db_ok,
    dataset_loaded=loaded,
    model_bundle_available=bundle_ok,
    latest_scoring_run_id=run_id,
)


@router.get("/me", response_model=s.Me, tags=["system"], summary="Current (development) identity")
def me(principal: Annotated[Principal, Depends(ANY_ROLE)]) -> s.Me:
    return s.Me(user_id=principal.user_id, role=principal.role, auth_mode=principal.auth_mode,
                display_name=principal.display_name)


@router.get("/dashboard/summary", response_model=s.DashboardSummary, tags=["dashboard"],
            summary="Cohort and worklist summary metrics")
def dashboard(session: SessionDep, _: Annotated[Principal, Depends(ANY_ROLE)]) -> s.DashboardSummary:
    load = latest_load(session)
    as_of = load.data_cutoff
    run = latest_run(session)

    patients_total = session.execute(select(func.count()).select_from(m.Patient)).scalar_one()
    patients_active = session.execute(select(func.count()).select_from(m.Patient).where(
        m.Patient.death_date.is_(None), m.Patient.transfer_out_date.is_(None))).scalar_one()
    open_q = (episode_base(m.FollowupEpisode.due_date, m.FollowupEpisode.window_end_date,
                           m.FollowupEpisode.is_cold_start, m.RiskScore.risk_tier)
              .outerjoin(m.RiskScore, and_(m.RiskScore.episode_id == m.FollowupEpisode.id,
                                           m.RiskScore.scoring_run_id == (run.id if run else -1)))
              .where(open_episode_filter()))
    open_rows = session.execute(open_q).all()
    due_dates = [r.due_date for r in open_rows]
    by_tier: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    predictive: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    upcoming = grace = overdue_unresolved = 0
    for r in open_rows:
        if r.risk_tier:
            by_tier[r.risk_tier] += 1
        if r.due_date > as_of:
            upcoming += 1
        elif r.window_end_date >= as_of:
            grace += 1
        else:
            overdue_unresolved += 1
        if r.risk_tier and r.window_end_date >= as_of:
            predictive[r.risk_tier] += 1
    weeks = []
    for w in range(8):
        start = as_of + timedelta(days=7 * w)
        weeks.append(s.WeekLoad(week_start=start, due_open_episodes=sum(
            1 for d in due_dates if start <= d < start + timedelta(days=7))))
    origins = dict(session.execute(select(m.OutreachAttempt.origin, func.count())
                                   .group_by(m.OutreachAttempt.origin)).all())
    end = datetime.combine(as_of + timedelta(days=1), time.min, tzinfo=UTC)
    last30 = session.execute(select(func.count()).select_from(m.OutreachAttempt).where(
        m.OutreachAttempt.attempted_at >= end - timedelta(days=30),
        m.OutreachAttempt.attempted_at < end)).scalar_one()
    lab = session.execute(
        select(m.FollowupEpisode.programme, m.EpisodeOutcome.label)
        .join(m.EpisodeOutcome, m.EpisodeOutcome.episode_id == m.FollowupEpisode.id)
        .where(m.EpisodeOutcome.label_status == "labelled",
               m.FollowupEpisode.window_end_date > as_of - timedelta(days=365),
               m.FollowupEpisode.window_end_date <= as_of)).all()
    by_prog = Counter(p for p, _ in lab)
    on_time = Counter(p for p, y in lab if y == 0)
    completion = {
        "window": f"label windows ending {as_of - timedelta(days=364)} to {as_of}",
        "labelled_episodes": len(lab),
        "completed_on_time_rate": (sum(on_time.values()) / len(lab)) if lab else None,
        "by_programme": {p: {"labelled": n, "completed_on_time_rate": on_time[p] / n}
                         for p, n in sorted(by_prog.items())},
    }
    return s.DashboardSummary(
        as_of=as_of, data_fingerprint=load.data_fingerprint, patients_total=patients_total,
        patients_active=patients_active, open_episodes=len(due_dates), open_by_tier=by_tier,
        open_unscored=len(due_dates) - sum(by_tier.values()),
        overdue_open_episodes=sum(1 for d in due_dates if d < as_of),
        due_next_14_days=sum(1 for d in due_dates if as_of <= d <= as_of + timedelta(days=14)),
        due_next_30_days=sum(1 for d in due_dates if as_of <= d <= as_of + timedelta(days=30)),
        due_by_week=weeks, outreach_historical_total=origins.get("historical_simulation", 0),
        outreach_app_total=origins.get("app", 0), outreach_last_30_days=last30,
        completion_last_12_months=completion, latest_scoring_run=run_out(run),
        open_upcoming=upcoming, open_in_grace_window=grace, open_overdue_unresolved=overdue_unresolved,
        predictive_by_tier=predictive,
        open_limited_history=sum(1 for r in open_rows if r.is_cold_start),
        notes=["All patients and events are synthetic.",
               "Tier counts are simulated predictions from the latest scoring run. predictive_by_tier "
               "counts only open follow-ups whose outcome window is still running; open_by_tier also "
               "includes overdue follow-ups whose score is historical.",
               f"Dates are relative to the dataset cutoff ({as_of}); outreach entered in the app is "
               "counted in outreach_app_total."])
