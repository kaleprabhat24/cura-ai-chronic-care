"""Patient onboarding: registration, later history entries and point-in-time scoring of new patients.

Everything is stored in the existing tables. The existing ML code is reused unchanged:
  * episodes  — `cura_ml.cohort.episodes.build_episodes` (programme intervals, 14-day lead,
                eligibility, censoring, maturity) with the saved bundle's own cohort config and
                *today* as the data cutoff;
  * features  — `cura_ml.features.build.build_features` on the patient's database rows, converted to
                the same columns as the dataset files (it filters every source on recorded time < t0);
  * model     — the saved bundle (preprocessing, calibrated logistic regression, tier thresholds) and
                its explanation code.

Point-in-time rules: clinical dates entered by the user are *effective* times. Every recorded time
(`recorded_at`, `booked_at`, `status_recorded_at`, `end_recorded_at`) is the real moment of entry and is
never backdated. A follow-up is scored only when its prediction time (due - 14 days) has arrived and
the history used was recorded before it. Otherwise the precise reason is returned and no number is
produced.

Identifiers follow the existing `<prefix>_app_<hex>` convention and are derived from the client's
submission id. A repeated submission therefore maps to the same primary keys and can never create a
second patient or duplicate history rows.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import uuid
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cura_api import models as m
from cura_api import schemas as s
from cura_api.api.common import PATIENT_SCOPE, effective_scores, is_app_patient, latest_run
from cura_api.auth import Principal
from cura_api.errors import ApiError
from cura_api.services import audit
from cura_api.services.scoring import bundle_metadata_sha256
from cura_api.services.seed import _hash, _link_outreach_to_episodes
from cura_ml.cohort.episodes import as_utc, build_episodes
from cura_ml.dataset import Records
from cura_ml.features import spec as S
from cura_ml.features.build import build_features
from cura_ml.modeling.bundle import ModelBundle, load_bundle
from cura_ml.synth import catalog as cat
from cura_ml.synth.generator import COLUMNS

log = logging.getLogger("cura_api.onboarding")
NS = uuid.UUID("2b8a7c41-5d0e-4f43-9b1c-7c3e1f5a9d20")
ENTRY_SOURCE = "cura_ai_entry"
PROGRAMME_ENCOUNTER = {"t2d": "review_t2d", "htn": "review_htn"}

# Database model -> dataset column renames (inverse of the seeding loader).
RECORD_SOURCES: dict[str, tuple[type[m.Base], dict[str, str]]] = {
    "patients": (m.Patient, {"id": "patient_id"}),
    "conditions": (m.Condition, {"id": "condition_id"}),
    "appointments": (m.Appointment, {"id": "appointment_id"}),
    "encounters": (m.Encounter, {"id": "encounter_id"}),
    "observations": (m.Observation, {"id": "observation_id"}),
    "medication_statements": (m.MedicationStatement, {"id": "medication_statement_id"}),
    "outreach_attempts": (m.OutreachAttempt, {"id": "outreach_id", "recorded_by": "performed_by"}),
}


def app_id(prefix: str, key: str, *parts: str) -> str:
    """Deterministic `<prefix>_app_<16 hex>` id from the submission id (+ item position)."""
    return f"{prefix}_app_{uuid.uuid5(NS, '|'.join((key.lower(), *parts))).hex[:16]}"


# ------------------------------------------------------------------------------ model + data access
@lru_cache(maxsize=2)
def _cached_bundle(path: str, mtime: float) -> ModelBundle:
    return load_bundle(path)


def get_bundle(bundle_dir: Path) -> ModelBundle:
    return _cached_bundle(str(bundle_dir), (bundle_dir / "metadata.json").stat().st_mtime)


def cohort_config_as_of(bundle: ModelBundle, day: date) -> dict[str, Any]:
    """The bundle's own cohort/episode rules, evaluated with `day` as the data cutoff."""
    cfg = copy.deepcopy(bundle.metadata["cohort_config"])
    cfg["simulation"]["data_cutoff"] = day.isoformat()
    return cfg


def records_from_db(session: Session, patient_id: str) -> Records:
    """One patient's persisted rows with exactly the dataset's column names and types."""
    frames: dict[str, pd.DataFrame] = {}
    for table, (model, rename) in RECORD_SOURCES.items():
        rows = (
            session.execute(
                select(model).where((model.id if model is m.Patient else model.patient_id) == patient_id)
            )
            .scalars()
            .all()
        )
        data = []
        for r in rows:
            d = {c.name: getattr(r, c.name) for c in model.__table__.columns}
            data.append({rename.get(k, k): v for k, v in d.items()})
        df = pd.DataFrame(data, columns=[rename.get(c.name, c.name) for c in model.__table__.columns])
        df = df.reindex(columns=COLUMNS[table])
        for col in df.columns:
            if col.endswith("_at") or col == "scheduled_for":
                df[col] = pd.to_datetime(df[col], utc=True)
        frames[table] = df
    return Records(**frames)


def _records_fingerprint(records: Records) -> str:
    h = hashlib.sha256()
    for t in RECORD_SOURCES:
        df = getattr(records, t)
        h.update(t.encode())
        h.update(df.sort_values(df.columns[0]).to_json(orient="records", date_format="iso").encode())
    return h.hexdigest()


def _set_hash(obj: m.Base) -> None:
    cols = [c.name for c in obj.__table__.columns if c.name != "row_hash"]
    obj.row_hash = _hash({c: getattr(obj, c) for c in cols}, cols)


# ------------------------------------------------------------------------------ history persistence
def _add_history(
    session: Session, patient_id: str, key: str, h: s.HistoryIn, now: datetime, actor: str, section: str
) -> dict[str, int]:
    for i, a in enumerate(h.appointments):
        obj = m.Appointment(
            id=app_id("ap", key, section, "appointment", str(i)),
            patient_id=patient_id,
            appointment_type=PROGRAMME_ENCOUNTER[a.programme],
            booked_at=now,  # when the booking became known to Cura AI
            scheduled_for=a.scheduled_for,
            status=a.status,
            status_recorded_at=now,
            booking_source=ENTRY_SOURCE,
        )
        _set_hash(obj)
        session.add(obj)
    for i, e in enumerate(h.encounters):
        obj = m.Encounter(
            id=app_id("en", key, section, "encounter", str(i)),
            patient_id=patient_id,
            encounter_type=e.encounter_type,
            start_at=e.start_at,
            end_at=e.start_at + timedelta(minutes=e.duration_minutes),
            status="finished",
            appointment_id=None,
            recorded_at=now,
        )
        _set_hash(obj)
        session.add(obj)
    for i, o in enumerate(h.observations):
        code, display, unit, _ = cat.OBSERVATIONS[o.kind]
        obj = m.Observation(
            id=app_id("ob", key, section, "observation", str(i)),
            patient_id=patient_id,
            encounter_id=None,
            code_system=cat.LOINC,
            code=code,
            display=display,
            value_num=float(o.value),
            unit=unit,
            effective_at=o.effective_at,
            recorded_at=now,
        )
        _set_hash(obj)
        session.add(obj)
    for i, x in enumerate(h.medications):
        code, display, drug_class = cat.MEDICATIONS[x.medication]
        obj = m.MedicationStatement(
            id=app_id("ms", key, section, "medication", str(i)),
            patient_id=patient_id,
            code_system=cat.RXNORM,
            code=code,
            display=display,
            drug_class=drug_class,
            start_date=x.start_date,
            end_date=x.end_date,
            recorded_at=now,
            end_recorded_at=now if x.end_date else None,
        )
        _set_hash(obj)
        session.add(obj)
    counts = {
        "appointments": len(h.appointments),
        "encounters": len(h.encounters),
        "observations": len(h.observations),
        "medications": len(h.medications),
    }
    outreach = getattr(h, "outreach", [])
    for i, o in enumerate(outreach):
        session.add(
            m.OutreachAttempt(
                id=app_id("oa", key, section, "outreach", str(i)),
                patient_id=patient_id,
                episode_id=None,
                programme=o.programme,
                origin="app",
                purpose="history_entry",
                channel=o.channel,
                outcome=o.outcome,
                note=o.note,
                next_action=o.next_action,
                next_action_due=o.next_action_due,
                attempted_at=o.attempted_at,
                recorded_at=now,
                recorded_by=actor,
                row_hash=None,
            )
        )
    counts["outreach"] = len(outreach)
    return counts


def refresh_episodes(
    session: Session, patient_id: str, bundle: ModelBundle, day: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild a registered patient's follow-up episodes with the existing rules and upsert the derived
    rows. Only patients registered in the app are refreshed; the synthetic cohort is never touched."""
    if not is_app_patient(patient_id):
        raise ValueError("episodes are only refreshed for patients registered in the app")
    records = records_from_db(session, patient_id)
    episodes, labels = build_episodes(records, cohort_config_as_of(bundle, day))
    keep = set()
    for row in episodes.to_dict("records"):
        values = {k: _py(v) for k, v in row.items()}
        values["id"] = values.pop("episode_id")
        keep.add(values["id"])
        obj = session.get(m.FollowupEpisode, values["id"]) or m.FollowupEpisode(id=values["id"])
        for k, v in values.items():
            setattr(obj, k, v)
        _set_hash(obj)
        session.add(obj)
    session.flush()
    for row in labels.to_dict("records"):
        values = {k: _py(v) for k, v in row.items()}
        obj = session.get(m.EpisodeOutcome, values["episode_id"]) or m.EpisodeOutcome(
            episode_id=values["episode_id"]
        )
        for k, v in values.items():
            setattr(obj, k, v)
        _set_hash(obj)
        session.add(obj)
    for old in session.execute(
        select(m.FollowupEpisode).where(
            m.FollowupEpisode.patient_id == patient_id, m.FollowupEpisode.id.not_in(keep)
        )
    ).scalars():
        old.eligible, old.ineligible_reason = False, "superseded_by_new_history"
        _set_hash(old)
    session.flush()
    _link_unlinked_outreach(session, patient_id, episodes)
    return episodes, labels


def _link_unlinked_outreach(session: Session, patient_id: str, episodes: pd.DataFrame) -> None:
    rows = (
        session.execute(
            select(m.OutreachAttempt).where(
                m.OutreachAttempt.patient_id == patient_id, m.OutreachAttempt.episode_id.is_(None)
            )
        )
        .scalars()
        .all()
    )
    if not rows or episodes.empty:
        return
    frame = pd.DataFrame(
        [
            {
                "outreach_id": r.id,
                "patient_id": r.patient_id,
                "programme": r.programme,
                "attempted_at": r.attempted_at,
            }
            for r in rows
        ]
    )
    links = _link_outreach_to_episodes(frame, episodes)
    for r in rows:
        ep = links.get(r.id)
        if isinstance(ep, str):
            r.episode_id = ep


def _py(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


# ------------------------------------------------------------------------------ scoring
REASONS = {
    "prediction_after_cutoff": "prediction_time_not_reached",
    "no_prior_encounter": "history_recorded_after_prediction_time",
    # for patients entered in Cura AI the condition is recorded at entry like the rest of the history
    "condition_not_recorded": "history_recorded_after_prediction_time",
    "completed_before_prediction": "completed_before_prediction",
    "prediction_before_study_period": "outside_study_period",
    "deceased_before_prediction": "patient_left_care",
    "transferred_before_prediction": "patient_left_care",
    "superseded_by_new_history": "superseded",
}


def _fmt(d: date | datetime) -> str:
    return (d.date() if isinstance(d, datetime) else d).strftime("%d %b %Y")


def _message(reason: str, row: dict[str, Any], first_recorded: datetime | None) -> str:
    t0, due = row["prediction_at"], row["due_date"]
    return {
        "prediction_time_not_reached": f"The prediction time is {_fmt(t0)}, 14 days before the {_fmt(due)} due date. The saved model "
        "scores a follow-up only from that date, using history recorded before it. Re-check then.",
        "history_recorded_after_prediction_time": f"This follow-up's prediction time ({_fmt(t0)}) had already passed when the patient's history was "
        f"recorded in Cura AI ({_fmt(first_recorded) if first_recorded else 'later'}). Records are not "
        "backdated, so no point-in-time prediction can be made for it. Record the next completed review "
        "when it happens; the following follow-up becomes scorable at its own prediction time.",
        "condition_recorded_after_prediction_time": f"The condition was recorded after this follow-up's prediction time ({_fmt(t0)}), so the model "
        "cannot use it for this follow-up.",
        "completed_before_prediction": "This follow-up was completed before its prediction time, so no prediction is needed.",
        "outside_study_period": f"The prediction time ({_fmt(t0)}) is before 2020-01-01, outside the model's study period.",
        "patient_left_care": "The patient left care before the prediction time.",
        "superseded": "Newer history changed this follow-up; see the current follow-up instead.",
    }[reason]


def _rank_in_batch(session: Session, probability: float) -> int:
    batch = latest_run(session)
    if batch is None:
        return 1
    higher = session.execute(
        select(func.count())
        .select_from(m.RiskScore)
        .where(m.RiskScore.scoring_run_id == batch.id, m.RiskScore.probability > probability)
    ).scalar_one()
    return int(higher) + 1


def score_patient(
    session: Session,
    patient_id: str,
    bundle: ModelBundle,
    bundle_dir: Path,
    now: datetime,
    actor: Principal | None,
    persist: bool,
) -> list[s.ScoreAvailability]:
    """Evaluate (and, with persist=True, score) a registered patient's follow-ups at time `now`.

    Scores every eligible, not-yet-scored episode whose prediction time has arrived, using the same
    eligibility rules, features, model and thresholds as cohort scoring. Never scores anything else.
    """
    day = now.date()
    cfg = cohort_config_as_of(bundle, day)
    records = records_from_db(session, patient_id)
    episodes, _ = build_episodes(records, cfg)
    if not episodes.empty:
        episodes["prediction_at"] = as_utc(episodes["prediction_at"])
    scored_ids = set(
        session.execute(select(m.RiskScore.episode_id).where(m.RiskScore.patient_id == patient_id)).scalars()
    )
    created: dict[str, m.RiskScore] = {}
    if persist and not episodes.empty:
        targets = episodes[
            episodes["eligible"]
            & ~episodes["episode_id"].isin(scored_ids)
            & (episodes["prediction_at"] <= pd.Timestamp(now))
        ]
        if len(targets):
            feats = build_features(records, targets, cfg, episodes)
            pred = bundle.predict(feats)
            explanations = bundle.explain(feats, top_k=5)
            md = bundle.metadata
            run = m.ScoringRun(
                status="succeeded",
                scope=PATIENT_SCOPE,
                model_version=bundle.version,
                model_type=md["model_type"],
                bundle_path=str(bundle_dir),
                bundle_metadata_sha256=bundle_metadata_sha256(bundle_dir),
                feature_spec_version=md["feature_spec_version"],
                calibration_method=md["calibration_method"],
                tier_thresholds=md["tier_thresholds"],
                data_fingerprint=_records_fingerprint(records),
                data_cutoff=day,
                started_at=now,
                finished_at=now,
                n_scored=len(pred),
                tier_counts={t: int((pred["risk_tier"] == t).sum()) for t in set(pred["risk_tier"])},
                triggered_by=actor.user_id if actor else "system",
                error=None,
            )
            session.add(run)
            session.flush()
            cols = bundle.feature_columns
            for i, (p, e) in enumerate(zip(pred.itertuples(index=False), explanations, strict=True)):
                snap = {c: (None if pd.isna(v) else float(v)) for c, v in feats.iloc[i][cols].items()}
                rs = m.RiskScore(
                    scoring_run_id=run.id,
                    episode_id=p.episode_id,
                    patient_id=patient_id,
                    model_version=bundle.version,
                    probability=float(p.probability),
                    model_log_odds=float(p.model_log_odds),
                    risk_tier=p.risk_tier,
                    rank=_rank_in_batch(session, float(p.probability)),
                    prediction_at=as_utc(feats["prediction_at"]).iloc[i].to_pydatetime(),
                    scored_at=now,
                    explanation=e,
                    feature_snapshot=snap,
                )
                session.add(rs)
                created[p.episode_id] = rs
            session.flush()
            audit.record(
                session,
                actor.user_id if actor else "system",
                actor.role if actor else "system",
                "scoring.patient",
                "scoring_run",
                str(run.id),
                {
                    "patient_id": patient_id,
                    "episodes": list(created),
                    "model_version": bundle.version,
                    "tier_counts": run.tier_counts,
                    "feature_spec_version": S.FEATURE_SPEC_VERSION,
                },
            )

    first_recorded = min(
        (
            ts
            for t in ("encounters", "appointments", "observations", "conditions")
            for ts in as_utc(
                getattr(records, t)["recorded_at" if t != "appointments" else "booked_at"]
            ).dropna()
        ),
        default=None,
    )
    programmes = sorted(
        set(records.conditions["code"].map({c["code"]: k for k, c in cat.CONDITIONS.items()}).dropna())
    )
    existing = effective_scores(session, list(episodes["episode_id"])) if not episodes.empty else {}
    out: list[s.ScoreAvailability] = []
    for prog in programmes:
        prog_eps = (
            episodes[episodes["programme"] == prog].sort_values("index_event_at")
            if not episodes.empty
            else episodes
        )
        if prog_eps.empty:
            out.append(
                s.ScoreAvailability(
                    programme=prog,
                    episode_id=None,
                    due_date=None,
                    prediction_at=None,
                    status="unavailable",
                    reason="no_review_history",
                    message="No completed review is recorded for this condition, so no follow-up can be scheduled "
                    "yet. Add the most recent completed review (a review visit or an attended review "
                    "appointment); the next due date follows from it "
                    f"({cfg['programmes'][prog]['interval_days']} days later) and the prediction time is "
                    "14 days before that.",
                )
            )
            continue
        row = prog_eps.iloc[-1].to_dict()
        base = {
            "programme": prog,
            "episode_id": row["episode_id"],
            "due_date": _py(row["due_date"]),
            "prediction_at": _py(row["prediction_at"]),
        }
        score = created.get(row["episode_id"]) or existing.get(row["episode_id"])
        if score is not None:
            new = row["episode_id"] in created
            note = (
                " Limited history (< 180 days) at prediction time."
                if row.get("is_cold_start") is True
                else ""
            )
            out.append(
                s.ScoreAvailability(
                    **base,
                    status="scored" if new else "already_scored",
                    message=(
                        "Scored with the saved model at its prediction time using history recorded before it."
                        if new
                        else "This follow-up already has a stored score; scores are never overwritten."
                    )
                    + note,
                    risk_score_id=score.id,
                    probability=score.probability,
                    risk_tier=score.risk_tier,
                    model_version=score.model_version,
                    scored_at=score.scored_at,
                )
            )
        elif row["eligible"]:
            out.append(
                s.ScoreAvailability(
                    **base,
                    status="ready_to_score",
                    message="The prediction time has arrived and the history recorded before it qualifies. Use "
                    "'Generate score' to score it with the saved model.",
                )
            )
        else:
            reason = REASONS.get(row["ineligible_reason"], "superseded")
            out.append(
                s.ScoreAvailability(
                    **base, status="unavailable", reason=reason, message=_message(reason, row, first_recorded)
                )
            )
    return out


# ------------------------------------------------------------------------------ public operations
def _episode_summaries(episodes: pd.DataFrame) -> list[s.OnboardingEpisode]:
    return [
        s.OnboardingEpisode(
            episode_id=r["episode_id"],
            programme=r["programme"],
            index_event_at=_py(r["index_event_at"]),
            due_date=_py(r["due_date"]),
            prediction_at=_py(r["prediction_at"]),
            eligible=bool(r["eligible"]),
            ineligible_reason=_py(r["ineligible_reason"]),
        )
        for r in episodes.to_dict("records")
    ]


def _score_safely(
    session: Session, patient_id: str, bundle: ModelBundle, bundle_dir: Path, now: datetime, actor: Principal
) -> list[s.ScoreAvailability]:
    try:
        statuses = score_patient(session, patient_id, bundle, bundle_dir, now, actor, persist=True)
        session.commit()
        return statuses
    except Exception:  # noqa: BLE001 - registration is already committed; report, do not fabricate
        session.rollback()
        log.exception("scoring failed for %s", patient_id)
        return [
            s.ScoreAvailability(
                programme="all",
                episode_id=None,
                due_date=None,
                prediction_at=None,
                status="unavailable",
                reason="scoring_error",
                message="The patient was saved, but scoring failed. Use 'Generate score' "
                "on the patient page to retry.",
            )
        ]


def _replay(
    session: Session,
    patient: m.Patient,
    key: str,
    action: str,
    bundle: ModelBundle,
    bundle_dir: Path,
    now: datetime,
) -> s.OnboardingResult:
    event = (
        session.execute(
            select(m.AuditEvent)
            .where(m.AuditEvent.action == action, m.AuditEvent.entity_id == patient.id)
            .order_by(m.AuditEvent.id.desc())
        )
        .scalars()
        .all()
    )
    match = next((e for e in event if (e.details or {}).get("submission_id") == key), None)
    counts = (match.details or {}).get("history_counts", {}) if match else {}
    episodes = (
        session.execute(
            select(m.FollowupEpisode)
            .where(m.FollowupEpisode.patient_id == patient.id)
            .order_by(m.FollowupEpisode.index_event_at)
        )
        .scalars()
        .all()
    )
    statuses = (
        score_patient(session, patient.id, bundle, bundle_dir, now, None, persist=False)
        if is_app_patient(patient.id)
        else []
    )
    return s.OnboardingResult(
        patient_id=patient.id,
        synthetic_mrn=patient.synthetic_mrn,
        created=False,
        history_counts=counts,
        recorded_at=match.occurred_at if match else now,
        episodes=[
            s.OnboardingEpisode(
                episode_id=e.id,
                programme=e.programme,
                index_event_at=e.index_event_at,
                due_date=e.due_date,
                prediction_at=e.prediction_at,
                eligible=e.eligible,
                ineligible_reason=e.ineligible_reason,
            )
            for e in episodes
        ],
        score_status=statuses,
        notes=["This submission was already processed; nothing was created twice."],
    )


def register_patient(
    session: Session, body: s.PatientRegistration, principal: Principal, now: datetime, bundle_dir: Path
) -> s.OnboardingResult:
    bundle = get_bundle(bundle_dir)
    key = body.submission_id.lower()
    pid = app_id("pt", key)
    existing = session.get(m.Patient, pid)
    if existing is not None:
        return _replay(session, existing, key, "patient.register", bundle, bundle_dir, now)
    if not body.allow_possible_duplicate:
        twin = (
            session.execute(
                select(m.Patient).where(
                    m.Patient.id.like("pt\\_app\\_%", escape="\\"),
                    func.lower(m.Patient.given_name) == body.given_name.strip().lower(),
                    func.lower(m.Patient.family_name) == body.family_name.strip().lower(),
                    m.Patient.birth_date == body.birth_date,
                )
            )
            .scalars()
            .first()
        )
        if twin is not None:
            raise ApiError(
                409,
                "possible_duplicate",
                "A registered patient with the same name and date of birth already exists.",
                {"patient_id": twin.id, "synthetic_mrn": twin.synthetic_mrn},
            )
    try:
        patient = m.Patient(
            id=pid,
            synthetic_mrn=f"SYNAPP{uuid.uuid5(NS, key).hex[:10].upper()}",
            given_name=body.given_name.strip(),
            family_name=body.family_name.strip(),
            birth_date=body.birth_date,
            sex=body.sex,
            deprivation_quintile=body.deprivation_quintile,
            preferred_language=body.preferred_language,
            registered_on=body.registered_on,
            death_date=None,
            death_recorded_at=None,
            transfer_out_date=None,
            transfer_recorded_at=None,
            is_synthetic=True,
        )
        _set_hash(patient)
        session.add(patient)
        session.flush()  # tables have no ORM relationships, so insert the parent row first
        for c in body.conditions:
            info = cat.CONDITIONS[c.programme]
            cond = m.Condition(
                id=app_id("cd", key, "condition", c.programme),
                patient_id=pid,
                code_system=cat.SNOMED,
                code=info["code"],
                display=info["display"],
                onset_date=c.onset_date,
                recorded_at=now,
                clinical_status="active",
            )
            _set_hash(cond)
            session.add(cond)
        session.flush()
        counts = _add_history(session, pid, key, body.history, now, principal.user_id, "registration")
        session.flush()
        episodes, _ = refresh_episodes(session, pid, bundle, now.date())
        audit.record(
            session,
            principal.user_id,
            principal.role,
            "patient.register",
            "patient",
            pid,
            {
                "submission_id": key,
                "history_counts": counts,
                "programmes": [c.programme for c in body.conditions],
                "episodes": len(episodes),
            },
        )
        session.commit()
    except IntegrityError:
        session.rollback()
        again = session.get(m.Patient, pid)
        if again is not None:  # a concurrent duplicate of this submission won the race
            return _replay(session, again, key, "patient.register", bundle, bundle_dir, now)
        raise ApiError(409, "conflict", "The registration conflicts with existing records.") from None
    except Exception:
        session.rollback()
        raise
    statuses = _score_safely(session, pid, bundle, bundle_dir, now, principal)
    return s.OnboardingResult(
        patient_id=pid,
        synthetic_mrn=patient.synthetic_mrn,
        created=True,
        history_counts=counts,
        recorded_at=now,
        episodes=_episode_summaries(episodes),
        score_status=statuses,
        notes=[
            "Clinical dates are stored as entered; every recorded time is the moment of entry (not backdated).",
            "Risk scores are simulated predictions from the saved model and are produced only at a follow-up's "
            "prediction time (14 days before its due date).",
        ],
    )


def add_history(
    session: Session,
    patient_id: str,
    body: s.HistoryAddition,
    principal: Principal,
    now: datetime,
    bundle_dir: Path,
) -> s.OnboardingResult:
    bundle = get_bundle(bundle_dir)
    patient = session.get(m.Patient, patient_id)
    if patient is None:
        raise ApiError(404, "not_found", f"Patient '{patient_id}' not found.")
    key = body.submission_id.lower()
    marker = f"history:{key}"
    first = _first_id(patient_id, key, body.history)
    if first and _exists(session, *first):
        return _replay(session, patient, marker, "patient.history_add", bundle, bundle_dir, now)
    codes = {
        c.code
        for c in session.execute(select(m.Condition).where(m.Condition.patient_id == patient_id)).scalars()
    }
    programmes = {k for k, c in cat.CONDITIONS.items() if c["code"] in codes}
    try:
        s.check_history_against(body.history, programmes, patient.birth_date)
    except ValueError as exc:
        raise ApiError(422, "validation_error", str(exc)) from None
    app_patient = is_app_patient(patient_id)
    try:
        counts = _add_history(
            session, patient_id, key, body.history, now, principal.user_id, f"history:{patient_id}"
        )
        session.flush()
        episodes = (
            refresh_episodes(session, patient_id, bundle, now.date())[0] if app_patient else pd.DataFrame()
        )
        audit.record(
            session,
            principal.user_id,
            principal.role,
            "patient.history_add",
            "patient",
            patient_id,
            {"submission_id": marker, "history_counts": counts},
        )
        session.commit()
    except IntegrityError:
        session.rollback()
        return _replay(session, patient, marker, "patient.history_add", bundle, bundle_dir, now)
    except Exception:
        session.rollback()
        raise
    notes = ["History rows were added; existing records and earlier risk scores are unchanged."]
    if app_patient:
        statuses = _score_safely(session, patient_id, bundle, bundle_dir, now, principal)
        summaries = _episode_summaries(episodes)
    else:
        statuses, summaries = [], []
        notes.append(
            "This patient belongs to the synthetic cohort, whose follow-ups and scores are fixed at the "
            "dataset cutoff; the new entries appear in the timeline but do not change those scores."
        )
    return s.OnboardingResult(
        patient_id=patient_id,
        synthetic_mrn=patient.synthetic_mrn,
        created=True,
        history_counts=counts,
        recorded_at=now,
        episodes=summaries,
        score_status=statuses,
        notes=notes,
    )


def _first_id(patient_id: str, key: str, h: s.HistoryIn) -> tuple[type[m.Base], str] | None:
    section = f"history:{patient_id}"
    for model, prefix, name, items in (
        (m.Appointment, "ap", "appointment", h.appointments),
        (m.Encounter, "en", "encounter", h.encounters),
        (m.Observation, "ob", "observation", h.observations),
        (m.MedicationStatement, "ms", "medication", h.medications),
    ):
        if items:
            return model, app_id(prefix, key, section, name, "0")
    return None


def _exists(session: Session, model: type[m.Base], id_: str) -> bool:
    return session.get(model, id_) is not None


def generate_scores(
    session: Session, patient_id: str, principal: Principal, now: datetime, bundle_dir: Path
) -> s.ScoreStatusOut:
    if not is_app_patient(patient_id):
        raise ApiError(409, "conflict", "Synthetic cohort patients are scored by cohort-wide scoring runs.")
    if session.get(m.Patient, patient_id) is None:
        raise ApiError(404, "not_found", f"Patient '{patient_id}' not found.")
    bundle = get_bundle(bundle_dir)
    try:
        refresh_episodes(session, patient_id, bundle, now.date())
        session.commit()
    except Exception:
        session.rollback()
        raise
    statuses = _score_safely(session, patient_id, bundle, bundle_dir, now, principal)
    return s.ScoreStatusOut(
        patient_id=patient_id, registered_in_app=True, evaluated_at=now, score_status=statuses
    )


def score_status(session: Session, patient_id: str, now: datetime, bundle_dir: Path) -> s.ScoreStatusOut:
    """Read-only evaluation: never writes, never scores."""
    patient = session.get(m.Patient, patient_id)
    if patient is None:
        raise ApiError(404, "not_found", f"Patient '{patient_id}' not found.")
    if not is_app_patient(patient_id):
        return s.ScoreStatusOut(
            patient_id=patient_id, registered_in_app=False, evaluated_at=now, score_status=[]
        )
    statuses = score_patient(
        session, patient_id, get_bundle(bundle_dir), bundle_dir, now, None, persist=False
    )
    session.rollback()
    return s.ScoreStatusOut(
        patient_id=patient_id, registered_in_app=True, evaluated_at=now, score_status=statuses
    )
