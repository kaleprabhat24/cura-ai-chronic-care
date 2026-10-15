"""ORM models.

Information classes are kept apart so the frontend can always label them:
  * recorded synthetic clinical history — patients, conditions, appointments, encounters,
    observations, medication statements, follow-up episodes (definitions known at t0);
  * recorded synthetic outcomes        — `episode_outcomes` (what happened after t0, as of the cutoff);
  * recorded outreach outcomes         — `outreach_attempts` (historical simulated + entered in the app);
  * simulated predictions              — `scoring_runs` / `risk_scores` (append-only snapshots).

Seeded tables carry `row_hash` so the loader can tell inserted / updated / unchanged rows apart.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from cura_api.db import Base, UTCDateTime

PROVENANCE_CLINICAL = "recorded_synthetic_clinical"
PROVENANCE_OUTCOME = "recorded_synthetic_outcome"
PROVENANCE_OUTREACH = "recorded_outreach_outcome"
PROVENANCE_PREDICTION = "simulated_prediction"


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    synthetic_mrn: Mapped[str] = mapped_column(String(32), unique=True)
    given_name: Mapped[str] = mapped_column(String(80))
    family_name: Mapped[str] = mapped_column(String(80), index=True)
    birth_date: Mapped[date] = mapped_column(Date)
    sex: Mapped[str] = mapped_column(String(16))
    deprivation_quintile: Mapped[int] = mapped_column(Integer)
    preferred_language: Mapped[str] = mapped_column(String(8))
    registered_on: Mapped[date] = mapped_column(Date)
    death_date: Mapped[date | None] = mapped_column(Date)
    death_recorded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    transfer_out_date: Mapped[date | None] = mapped_column(Date)
    transfer_recorded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    row_hash: Mapped[str] = mapped_column(String(32))


class Condition(Base):
    __tablename__ = "conditions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    code_system: Mapped[str] = mapped_column(String(80))
    code: Mapped[str] = mapped_column(String(32))
    display: Mapped[str] = mapped_column(String(160))
    onset_date: Mapped[date] = mapped_column(Date)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    clinical_status: Mapped[str] = mapped_column(String(16))
    row_hash: Mapped[str] = mapped_column(String(32))


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    appointment_type: Mapped[str] = mapped_column(String(32))
    booked_at: Mapped[datetime] = mapped_column(UTCDateTime)
    scheduled_for: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(24))
    status_recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    booking_source: Mapped[str] = mapped_column(String(32))
    row_hash: Mapped[str] = mapped_column(String(32))


class Encounter(Base):
    __tablename__ = "encounters"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    encounter_type: Mapped[str] = mapped_column(String(32))
    start_at: Mapped[datetime] = mapped_column(UTCDateTime)
    end_at: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16))
    appointment_id: Mapped[str | None] = mapped_column(ForeignKey("appointments.id"))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    row_hash: Mapped[str] = mapped_column(String(32))


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    encounter_id: Mapped[str | None] = mapped_column(ForeignKey("encounters.id"))
    code_system: Mapped[str] = mapped_column(String(80))
    code: Mapped[str] = mapped_column(String(32))
    display: Mapped[str] = mapped_column(String(160))
    value_num: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))
    effective_at: Mapped[datetime] = mapped_column(UTCDateTime)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    row_hash: Mapped[str] = mapped_column(String(32))


class MedicationStatement(Base):
    __tablename__ = "medication_statements"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    code_system: Mapped[str] = mapped_column(String(80))
    code: Mapped[str] = mapped_column(String(32))
    display: Mapped[str] = mapped_column(String(160))
    drug_class: Mapped[str] = mapped_column(String(48))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    end_recorded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    row_hash: Mapped[str] = mapped_column(String(32))


class FollowupEpisode(Base):
    """Episode definition and eligibility: information known at the prediction timestamp."""

    __tablename__ = "followup_episodes"
    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    programme: Mapped[str] = mapped_column(String(8), index=True)
    condition_code: Mapped[str] = mapped_column(String(32))
    index_event_at: Mapped[datetime] = mapped_column(UTCDateTime)
    index_event_type: Mapped[str] = mapped_column(String(16))
    index_event_ref: Mapped[str] = mapped_column(String(40))
    due_date: Mapped[date] = mapped_column(Date, index=True)
    prediction_at: Mapped[datetime] = mapped_column(UTCDateTime)
    window_end_date: Mapped[date] = mapped_column(Date)
    eligible: Mapped[bool] = mapped_column(Boolean, index=True)
    ineligible_reason: Mapped[str | None] = mapped_column(String(48))
    is_cold_start: Mapped[bool | None] = mapped_column(Boolean)
    history_days: Mapped[int | None] = mapped_column(Integer)
    row_hash: Mapped[str] = mapped_column(String(32))


class EpisodeOutcome(Base):
    """What happened after t0, as recorded by the data cutoff (synthetic)."""

    __tablename__ = "episode_outcomes"
    episode_id: Mapped[str] = mapped_column(ForeignKey("followup_episodes.id"), primary_key=True)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    completion_event_type: Mapped[str | None] = mapped_column(String(16))
    completion_event_ref: Mapped[str | None] = mapped_column(String(40))
    completed_in_window: Mapped[bool | None] = mapped_column(Boolean)
    censored: Mapped[bool] = mapped_column(Boolean)
    censor_reason: Mapped[str | None] = mapped_column(String(16))
    censor_date: Mapped[date | None] = mapped_column(Date)
    mature: Mapped[bool] = mapped_column(Boolean)
    label_status: Mapped[str] = mapped_column(String(16), index=True)
    label: Mapped[int | None] = mapped_column(Integer)
    row_hash: Mapped[str] = mapped_column(String(32))


class OutreachAttempt(Base):
    __tablename__ = "outreach_attempts"
    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    episode_id: Mapped[str | None] = mapped_column(ForeignKey("followup_episodes.id"), index=True)
    programme: Mapped[str] = mapped_column(String(8))
    origin: Mapped[str] = mapped_column(String(24))  # historical_simulation | app
    purpose: Mapped[str] = mapped_column(String(32))
    channel: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[str] = mapped_column(String(24), index=True)
    note: Mapped[str | None] = mapped_column(Text)
    next_action: Mapped[str] = mapped_column(String(32))
    next_action_due: Mapped[date | None] = mapped_column(Date)
    attempted_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    recorded_by: Mapped[str] = mapped_column(String(80))
    row_hash: Mapped[str | None] = mapped_column(String(32))


class DatasetLoad(Base):
    """One row per seed run: which dataset (fingerprint, cutoff) the database reflects."""

    __tablename__ = "dataset_loads"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_version: Mapped[str] = mapped_column(String(16))
    data_fingerprint: Mapped[str] = mapped_column(String(64))
    data_cutoff: Mapped[date] = mapped_column(Date)
    dataset_dir: Mapped[str] = mapped_column(String(400))
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    counts: Mapped[dict] = mapped_column(JSON)


class ScoringRun(Base):
    __tablename__ = "scoring_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(16))  # succeeded | failed
    scope: Mapped[str] = mapped_column(String(16))   # eligible | open
    model_version: Mapped[str] = mapped_column(String(40), index=True)
    model_type: Mapped[str] = mapped_column(String(32))
    bundle_path: Mapped[str] = mapped_column(String(400))
    bundle_metadata_sha256: Mapped[str] = mapped_column(String(64))
    feature_spec_version: Mapped[str] = mapped_column(String(16))
    calibration_method: Mapped[str] = mapped_column(String(16))
    tier_thresholds: Mapped[dict] = mapped_column(JSON)
    data_fingerprint: Mapped[str] = mapped_column(String(64))
    data_cutoff: Mapped[date] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    n_scored: Mapped[int] = mapped_column(Integer, default=0)
    tier_counts: Mapped[dict] = mapped_column(JSON, default=dict)
    triggered_by: Mapped[str] = mapped_column(String(80))
    error: Mapped[str | None] = mapped_column(Text)


class RiskScore(Base):
    """One simulated prediction for one episode in one scoring run (never updated in place)."""

    __tablename__ = "risk_scores"
    __table_args__ = (Index("ix_risk_scores_run_episode", "scoring_run_id", "episode_id", unique=True),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scoring_run_id: Mapped[int] = mapped_column(ForeignKey("scoring_runs.id"), index=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("followup_episodes.id"), index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    probability: Mapped[float] = mapped_column(Float)
    model_log_odds: Mapped[float] = mapped_column(Float)
    risk_tier: Mapped[str] = mapped_column(String(8), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    prediction_at: Mapped[datetime] = mapped_column(UTCDateTime)
    scored_at: Mapped[datetime] = mapped_column(UTCDateTime)
    explanation: Mapped[dict] = mapped_column(JSON)
    feature_snapshot: Mapped[dict] = mapped_column(JSON)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    actor_id: Mapped[str] = mapped_column(String(80))
    actor_role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(48), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSON, default=dict)
