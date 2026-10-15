"""Outreach outcomes, model information, scoring runs, data quality and audit log."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select

from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import PageParams, SessionDep, latest_load, latest_run, run_out
from cura_api.auth import ADMIN, ANALYTICS, ANY_ROLE, Principal
from cura_api.errors import ERROR_RESPONSES, ApiError, ErrorResponse
from cura_ml.manifest import data_fingerprint, read_manifest

router = APIRouter(responses=ERROR_RESPONSES)
VERSION = re.compile(r"^[A-Za-z0-9._-]{1,60}$")
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Not found"}}
DISCLAIMER = ("Synthetic-data evaluation, not clinical validation. The model predicts follow-up "
              "non-completion; it does not predict deterioration or diagnose.")


# ------------------------------------------------------------------------------------------ outcomes
@router.get("/outreach/outcomes", response_model=s.OutcomesSummary, tags=["outreach"],
            summary="Aggregate outreach outcomes and follow-up completion statistics")
def outcomes(session: SessionDep, _: Annotated[Principal, Depends(ANY_ROLE)]) -> s.OutcomesSummary:
    as_of = latest_load(session).data_cutoff
    rows = session.execute(select(m.OutreachAttempt.origin, m.OutreachAttempt.outcome,
                                  m.OutreachAttempt.channel, func.count())
                           .group_by(m.OutreachAttempt.origin, m.OutreachAttempt.outcome,
                                     m.OutreachAttempt.channel)).all()
    by_origin: Counter[str] = Counter()
    by_outcome: dict[str, dict[str, int]] = defaultdict(dict)
    by_channel: Counter[str] = Counter()
    phone: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for origin, outcome, channel, n in rows:
        by_origin[origin] += n
        by_outcome[origin][outcome] = by_outcome[origin].get(outcome, 0) + n
        by_channel[channel] += n
        if channel == "phone":
            phone[origin][1] += n
            if outcome in s.REACHED:
                phone[origin][0] += n
    contact = {o: (r / t if t else None) for o, (r, t) in phone.items()}
    month_start = date(as_of.year - 2, as_of.month, 1)
    monthly: dict[tuple[str, str, str], int] = Counter()
    for when, origin, outcome, channel in session.execute(
            select(m.OutreachAttempt.attempted_at, m.OutreachAttempt.origin, m.OutreachAttempt.outcome,
                   m.OutreachAttempt.channel)
            .where(m.OutreachAttempt.attempted_at >= datetime.combine(month_start, time.min, tzinfo=UTC))):
        group = ("letter" if channel == "letter" else
                 "reached" if outcome in s.REACHED else "not_reached")
        monthly[(when.strftime("%Y-%m"), origin, group)] += 1
    attempts_by_month = [{"month": k[0], "origin": k[1], "group": k[2], "attempts": n}
                         for k, n in sorted(monthly.items())]
    open_actions = session.execute(
        select(func.count()).select_from(m.OutreachAttempt)
        .join(m.EpisodeOutcome, m.EpisodeOutcome.episode_id == m.OutreachAttempt.episode_id)
        .where(m.OutreachAttempt.origin == "app", m.OutreachAttempt.next_action != "none",
               m.EpisodeOutcome.completed_at.is_(None))).scalar_one()

    lab = session.execute(select(m.FollowupEpisode.id, m.FollowupEpisode.programme,
                                 m.FollowupEpisode.prediction_at, m.FollowupEpisode.window_end_date,
                                 m.EpisodeOutcome.label)
                          .join(m.EpisodeOutcome, m.EpisodeOutcome.episode_id == m.FollowupEpisode.id)
                          .where(m.EpisodeOutcome.label_status == "labelled")).all()
    reached_eps = set(session.execute(select(m.OutreachAttempt.episode_id).where(
        m.OutreachAttempt.channel == "phone", m.OutreachAttempt.episode_id.is_not(None))).scalars())

    def summarise(key) -> list[dict[str, Any]]:
        groups: dict[Any, list[int]] = defaultdict(lambda: [0, 0])
        for row in lab:
            g = groups[key(row)]
            g[0] += 1
            g[1] += row.label == 0
        return [{"group": k, "labelled": n, "completed_on_time_rate": c / n}
                for k, (n, c) in sorted(groups.items(), key=lambda kv: str(kv[0]))]

    return s.OutcomesSummary(
        as_of=as_of, attempts_by_month=attempts_by_month, attempts_by_origin=dict(by_origin),
        attempts_by_outcome=dict(by_outcome),
        attempts_by_channel=dict(by_channel), phone_contact_rate=contact, app_open_next_actions=open_actions,
        completion_by_year=summarise(lambda r: r.prediction_at.year),
        completion_by_programme=summarise(lambda r: r.programme),
        completion_by_outreach_exposure=summarise(
            lambda r: "phone outreach recorded" if r.id in reached_eps else "no phone outreach recorded"),
        notes=["Recorded outcomes of synthetic episodes; labelled episodes only (mature, uncensored).",
               "Completion by outreach exposure is descriptive, not causal: outreach was targeted at "
               "patients already overdue.",
               "Historical recall letters are stored with channel 'letter' and outcome 'other'."])


# ------------------------------------------------------------------------------------------ models
def _bundle_info(bundle_dir: Path) -> s.ModelInfo:
    if not (bundle_dir / "metadata.json").exists():
        raise ApiError(404, "not_found", f"No model bundle at '{bundle_dir.name}'.")
    md = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    ev = json.loads((bundle_dir / "evaluation.json").read_text(encoding="utf-8"))
    keys = ("pr_auc", "roc_auc", "brier", "ece", "calibration_slope", "prevalence", "n", "capacity")
    return s.ModelInfo(
        model_version=md["model_version"], model_type=md["model_type"],
        calibration_method=md["calibration_method"], feature_spec_version=md["feature_spec_version"],
        tier_thresholds=md["tier_thresholds"], data_fingerprint=md["data_fingerprint"],
        prediction_target=md["prediction_target"], trained_on=md["trained_on"],
        calibrated_on=md["calibrated_on"], selection=ev["selection"], splits=ev["splits"],
        test_metrics={k: {x: v[x] for x in keys} for k, v in ev["test"]["point"].items()},
        test_intervals=ev["test"]["bootstrap"]["models"],
        validation_metrics={k: {x: v[x] for x in keys} for k, v in ev["validation"].items()},
        library_versions=md["library_versions"],
        calibration_table=ev["test"].get("calibration_table", []),
        test_tiers=ev["test"].get("tiers", []),
        feature_importance=(ev.get("feature_importance", {}).get(md["model_type"], {})
                            .get("features", [])[:12]),
        synthetic_data_only=md["synthetic_data_only"], disclaimer=DISCLAIMER)


@router.get("/models/current", response_model=s.ModelInfo, tags=["model"],
            summary="Model version used for scoring, with evaluation results")
def model_current(request: Request, _: Annotated[Principal, Depends(ANY_ROLE)]) -> s.ModelInfo:
    return _bundle_info(request.app.state.settings.model_bundle_dir)


@router.get("/models/current/documents", response_model=s.ModelDocuments, tags=["model"],
            summary="Model card and evaluation report (Markdown) for the scoring model")
def model_documents(request: Request, _: Annotated[Principal, Depends(ANY_ROLE)]) -> s.ModelDocuments:
    settings = request.app.state.settings
    md_path = settings.model_bundle_dir / "metadata.json"
    if not md_path.exists():
        raise ApiError(404, "not_found", "No model bundle is configured.")
    version = json.loads(md_path.read_text("utf-8"))["model_version"]
    card = settings.model_bundle_dir / "model_card.md"
    report = settings.model_reports_dir / f"model_{version}.md"
    return s.ModelDocuments(model_version=version,
                            model_card_markdown=card.read_text("utf-8") if card.exists() else None,
                            model_report_markdown=report.read_text("utf-8") if report.exists() else None)


@router.get("/models/{model_version}", response_model=s.ModelInfo, tags=["model"],
            summary="A saved model version", responses=NOT_FOUND)
def model_version(model_version: str, request: Request,
                  _: Annotated[Principal, Depends(ANALYTICS)]) -> s.ModelInfo:
    if not VERSION.match(model_version):
        raise ApiError(422, "validation_error", "Invalid model version.")
    return _bundle_info(request.app.state.settings.model_artifacts_root / model_version)


# ------------------------------------------------------------------------------------------ scoring runs
@router.get("/scoring-runs", response_model=s.Page[s.ScoringRunOut], tags=["model"],
            summary="Scoring runs (newest first)")
def scoring_runs(session: SessionDep, _: Annotated[Principal, Depends(ANALYTICS)],
                 paging: Annotated[PageParams, Depends()]) -> s.Page[s.ScoringRunOut]:
    total = session.execute(select(func.count()).select_from(m.ScoringRun)).scalar_one()
    rows = session.execute(select(m.ScoringRun).order_by(m.ScoringRun.id.desc())
                           .offset(paging.offset).limit(paging.page_size)).scalars()
    return s.Page[s.ScoringRunOut](items=[run_out(r) for r in rows], **paging.meta(total))


@router.get("/scoring-runs/{run_id}", response_model=s.ScoringRunOut, tags=["model"],
            summary="One scoring run", responses=NOT_FOUND)
def scoring_run(run_id: int, session: SessionDep,
                _: Annotated[Principal, Depends(ANALYTICS)]) -> s.ScoringRunOut:
    run = session.get(m.ScoringRun, run_id)
    if run is None:
        raise ApiError(404, "not_found", f"Scoring run {run_id} not found.")
    return run_out(run)


# ------------------------------------------------------------------------------------------ data quality
@router.get("/data-quality", response_model=s.DataQuality, tags=["model"],
            summary="Dataset, database and scoring consistency checks")
def data_quality(request: Request, session: SessionDep,
                 _: Annotated[Principal, Depends(ANALYTICS)]) -> s.DataQuality:
    settings = request.app.state.settings
    load = session.execute(select(m.DatasetLoad).order_by(m.DatasetLoad.id.desc())).scalars().first()
    run = latest_run(session)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    manifest = None
    if (settings.dataset_dir / "manifest.json").exists():
        manifest = read_manifest(settings.dataset_dir)
    file_fp = data_fingerprint(manifest) if manifest else None
    check("dataset_loaded", load is not None,
          "cohort loaded into the database" if load else "run cura-api seed")
    check("database_matches_dataset_files", bool(load and file_fp and load.data_fingerprint == file_fp),
          "fingerprint of loaded data equals the dataset files on disk")
    bundle_md = settings.model_bundle_dir / "metadata.json"
    bundle_fp = json.loads(bundle_md.read_text("utf-8"))["data_fingerprint"] if bundle_md.exists() else None
    check("model_trained_on_loaded_dataset", bool(load and bundle_fp == load.data_fingerprint),
          "model bundle data fingerprint equals the loaded dataset")
    run_matches = bool(run and load and run.data_fingerprint == load.data_fingerprint)
    check("latest_scoring_run_matches_dataset", run_matches,
          "latest scoring run used the loaded dataset")
    n_eligible = session.execute(select(func.count()).select_from(m.FollowupEpisode).where(
        m.FollowupEpisode.eligible.is_(True))).scalar_one()
    scored = 0
    if run:
        scored = session.execute(select(func.count(func.distinct(m.RiskScore.episode_id)))
                                 .where(m.RiskScore.scoring_run_id == run.id)).scalar_one()
    expected = n_eligible if not run or run.scope == "eligible" else scored
    check("eligible_episodes_scored", bool(run) and scored == expected,
          f"{scored:,} of {expected:,} episodes in scope scored by the latest run")

    table_counts: dict[str, dict[str, int | None]] = {}
    tables = {"patients": m.Patient, "conditions": m.Condition, "appointments": m.Appointment,
              "encounters": m.Encounter, "observations": m.Observation,
              "medication_statements": m.MedicationStatement, "followup_episodes": m.FollowupEpisode,
              "episode_outcomes": m.EpisodeOutcome}
    source = {"followup_episodes": "cohort/followup_episodes.parquet",
              "episode_outcomes": "outcomes/episode_labels.parquet"}
    for name, model in tables.items():
        db_n = session.execute(select(func.count()).select_from(model)).scalar_one()
        key = source.get(name, f"records/{name}.parquet")
        src_n = manifest["files"][key]["rows"] if manifest and key in manifest["files"] else None
        table_counts[name] = {"database": db_n, "dataset_files": src_n}
        if src_n is not None:
            check(f"row_count_{name}", db_n == src_n, f"database {db_n:,} vs files {src_n:,}")
    hist = session.execute(select(func.count()).select_from(m.OutreachAttempt).where(
        m.OutreachAttempt.origin == "historical_simulation")).scalar_one()
    app = session.execute(select(func.count()).select_from(m.OutreachAttempt).where(
        m.OutreachAttempt.origin == "app")).scalar_one()
    src_o = manifest["files"]["records/outreach_attempts.parquet"]["rows"] if manifest else None
    table_counts["outreach_attempts"] = {"database_historical": hist, "database_app": app,
                                         "dataset_files": src_o}
    if src_o is not None:
        check("row_count_outreach_historical", hist == src_o, f"database {hist:,} vs files {src_o:,}")
    labels = dict(session.execute(select(m.EpisodeOutcome.label_status, func.count())
                                  .group_by(m.EpisodeOutcome.label_status)).all())
    missing: dict[str, float] = {}
    ev_path = settings.model_bundle_dir / "evaluation.json"
    if ev_path.exists():
        missing = json.loads(ev_path.read_text("utf-8")).get("feature_missingness", {}).get("test", {})
    return s.DataQuality(status="ok" if all(c["ok"] for c in checks) else "warning", checks=checks,
                         dataset=None if load is None else {
                             "dataset_version": load.dataset_version,
                             "data_fingerprint": load.data_fingerprint,
                             "data_cutoff": load.data_cutoff.isoformat(),
                             "loaded_at": load.loaded_at.isoformat(),
                             "last_seed_counts": load.counts},
                         table_counts=table_counts, label_status_counts=labels,
                         feature_missingness={k: v for k, v in missing.items() if v > 0},
                         latest_scoring_run=run_out(run))


# ------------------------------------------------------------------------------------------ audit
@router.get("/audit-events", response_model=s.Page[s.AuditEventOut], tags=["admin"],
            summary="Audit log (newest first)")
def audit_events(session: SessionDep, _: Annotated[Principal, Depends(ADMIN)],
                 paging: Annotated[PageParams, Depends()],
                 action: Annotated[str | None, Query(max_length=48)] = None) -> s.Page[s.AuditEventOut]:
    base = select(m.AuditEvent)
    if action:
        base = base.where(m.AuditEvent.action == action)
    total = session.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    rows = session.execute(base.order_by(m.AuditEvent.id.desc()).offset(paging.offset)
                           .limit(paging.page_size)).scalars()
    return s.Page[s.AuditEventOut](items=[s.AuditEventOut.model_validate(r) for r in rows],
                                   **paging.meta(total))


@router.get("/admin/users", response_model=list[s.AppUserOut], tags=["admin"],
            summary="Provisioned application users and roles")
def admin_users(session: SessionDep, _: Annotated[Principal, Depends(ADMIN)]) -> list[s.AppUserOut]:
    rows = session.execute(select(m.AppUser).order_by(m.AppUser.email)).scalars()
    return [s.AppUserOut(id=u.id, email=u.email, display_name=u.display_name, role=u.role,
                         is_active=u.is_active, linked_to_auth_identity=u.auth_user_id is not None,
                         created_at=u.created_at, last_sign_in_at=u.last_sign_in_at) for u in rows]
