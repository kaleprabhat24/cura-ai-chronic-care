"""Typed request/response schemas (the frontend-to-backend contract).

Every object that carries data states its `provenance`:
  simulated_prediction         — model output from a scoring run
  recorded_synthetic_clinical  — synthetic clinical / administrative history
  recorded_synthetic_outcome   — what happened to an episode, as recorded by the data cutoff
  recorded_outreach_outcome    — outreach attempts (historical simulation or entered in the app)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Generic, Literal, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T")

Outcome = Literal["no_answer", "left_message", "reached_booked", "reached_declined",
                  "reached_already_done", "wrong_number", "other"]
Channel = Literal["phone", "sms", "letter", "email", "in_person"]
NextAction = Literal["none", "retry", "retry_later", "book_appointment", "await_response",
                     "update_contact_details", "escalate", "other"]
Tier = Literal["high", "medium", "low"]
Programme = Literal["t2d", "htn"]
EventType = Literal["appointment", "encounter", "observation", "medication_start", "medication_stop",
                    "outreach", "risk_score", "followup_due"]
OutcomeStatus = Literal["open", "open_overdue", "completed_on_time", "completed_late", "censored",
                        "not_eligible"]
FollowUpStatus = Literal["upcoming", "due_in_grace_window", "overdue_unresolved", "completed", "censored",
                         "not_eligible"]
ScoreStatus = Literal["predictive", "historical", "no_score"]
WorklistQueue = Literal["all", "predictive", "overdue", "no_score"]
REACHED: tuple[str, ...] = ("reached_booked", "reached_declined", "reached_already_done")


class Page(BaseModel, Generic[T]):  # noqa: UP046 - explicit Generic form documented by pydantic
    items: list[T]
    page: int
    page_size: int
    total: int
    total_pages: int


# ---------------------------------------------------------------- shared pieces
class Factor(BaseModel):
    feature: str
    label: str
    value: float | str | None
    value_text: str
    unit: str
    recorded: bool
    contribution_log_odds: float
    direction: str
    window: str | None = Field(default=None, description="Look-back window of the feature at prediction time")
    source: str | None = None


class ScoreContext(BaseModel):
    """How to read a score for this episode at the data cutoff (`as_of`).

    * follow_up_status — recorded state of the follow-up: upcoming (before due date),
      due_in_grace_window (past due, 30-day completion window still open), overdue_unresolved (not
      completed by the end of the window: an observed outcome needing operational follow-up),
      completed, censored, not_eligible.
    * score_status — predictive (outcome not yet known, so the probability is a live prediction),
      historical (the outcome window has closed or the episode is resolved; the probability is a past
      prediction kept for provenance), no_score.
    """

    as_of: date
    follow_up_status: FollowUpStatus
    score_status: ScoreStatus
    outcome_window_ended: bool
    outcome_mature: bool
    days_overdue: int
    score_age_days: int | None
    data_sufficiency: Literal["adequate", "limited_history"]
    summary: str


class PatientSummary(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    synthetic_mrn: str
    display_name: str
    age_years: int
    sex: str


class PredictionSummary(BaseModel):
    provenance: Literal["simulated_prediction"] = "simulated_prediction"
    risk_score_id: int
    scoring_run_id: int
    probability: float = Field(description="Calibrated probability that follow-up is still "
                                           "incomplete 30 days after the due date")
    risk_tier: Tier
    rank: int
    model_version: str
    prediction_at: datetime
    scored_at: datetime
    top_factor: Factor | None


class OutreachSummary(BaseModel):
    provenance: Literal["recorded_outreach_outcome"] = "recorded_outreach_outcome"
    id: str
    attempted_at: datetime
    channel: str
    outcome: str
    origin: str


class RecordedOutcome(BaseModel):
    provenance: Literal["recorded_synthetic_outcome"] = "recorded_synthetic_outcome"
    status: OutcomeStatus
    completed_at: datetime | None
    completed_in_window: bool | None
    label_status: str
    censor_reason: str | None


# ---------------------------------------------------------------- health / me
class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    auth_mode: str
    production_auth_implemented: bool = False
    database: Literal["ok", "error"]
    dataset_loaded: bool
    model_bundle_available: bool
    latest_scoring_run_id: int | None


class Me(BaseModel):
    user_id: str
    role: str
    auth_mode: str
    display_name: str | None = None


# ---------------------------------------------------------------- scoring runs
class ScoringRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    scope: str
    model_version: str
    model_type: str
    bundle_metadata_sha256: str
    feature_spec_version: str
    calibration_method: str
    tier_thresholds: dict[str, float]
    data_fingerprint: str
    data_cutoff: date
    started_at: datetime
    finished_at: datetime | None
    n_scored: int
    tier_counts: dict[str, int]
    triggered_by: str
    error: str | None


# ---------------------------------------------------------------- dashboard
class WeekLoad(BaseModel):
    week_start: date
    due_open_episodes: int


class DashboardSummary(BaseModel):
    as_of: date = Field(description="Data cutoff of the loaded synthetic dataset ('simulated today')")
    data_fingerprint: str
    patients_total: int
    patients_active: int
    open_episodes: int
    open_by_tier: dict[str, int]
    open_unscored: int
    overdue_open_episodes: int
    due_next_14_days: int
    due_next_30_days: int
    due_by_week: list[WeekLoad]
    outreach_historical_total: int
    outreach_app_total: int
    outreach_last_30_days: int
    completion_last_12_months: dict[str, Any]
    latest_scoring_run: ScoringRunOut | None
    open_upcoming: int = 0
    open_in_grace_window: int = 0
    open_overdue_unresolved: int = 0
    predictive_by_tier: dict[str, int] = Field(
        default_factory=dict, description="Tiers of open episodes whose score is still predictive")
    open_limited_history: int = 0
    notes: list[str]


# ---------------------------------------------------------------- worklist
class WorklistItem(BaseModel):
    episode_id: str
    programme: str
    condition: str
    due_date: date
    days_until_due: int
    prediction_at: datetime
    patient: PatientSummary
    prediction: PredictionSummary | None
    last_outreach: OutreachSummary | None
    outreach_attempts: int
    recorded_outcome: RecordedOutcome
    score_context: ScoreContext


class WorklistPage(Page[WorklistItem]):
    as_of: date
    scoring_run_id: int | None


# ---------------------------------------------------------------- patient
class PatientOut(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    synthetic_mrn: str
    given_name: str
    family_name: str
    birth_date: date
    age_years: int
    sex: str
    deprivation_quintile: int
    preferred_language: str
    registered_on: date
    death_date: date | None
    transfer_out_date: date | None
    is_synthetic: bool


class ConditionOut(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    code_system: str
    code: str
    display: str
    onset_date: date
    recorded_at: datetime
    clinical_status: str


class MedicationOut(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    code: str
    display: str
    drug_class: str
    start_date: date
    end_date: date | None
    recorded_at: datetime


class ObservationOut(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    code: str
    display: str
    value: float
    unit: str
    effective_at: datetime
    recorded_at: datetime


class EpisodeOut(BaseModel):
    provenance: Literal["recorded_synthetic_clinical"] = "recorded_synthetic_clinical"
    id: str
    patient_id: str
    programme: str
    condition: str
    index_event_at: datetime
    due_date: date
    prediction_at: datetime
    window_end_date: date
    eligible: bool
    ineligible_reason: str | None
    is_cold_start: bool | None
    recorded_outcome: RecordedOutcome
    latest_prediction: PredictionSummary | None
    score_context: ScoreContext


class PatientDetail(BaseModel):
    as_of: date
    patient: PatientOut
    conditions: list[ConditionOut]
    episodes: list[EpisodeOut]
    active_medications: list[MedicationOut]
    latest_observations: list[ObservationOut]
    counts: dict[str, int]


class TimelineEvent(BaseModel):
    id: str
    type: EventType
    occurred_at: datetime
    recorded_at: datetime | None
    title: str
    detail: dict[str, Any]
    provenance: str
    episode_id: str | None = None


class Timeline(BaseModel):
    patient_id: str
    total: int
    returned: int
    items: list[TimelineEvent]


# ---------------------------------------------------------------- risk score
class FeatureValue(BaseModel):
    column: str
    label: str
    unit: str
    value: float | None


class Explanation(BaseModel):
    method: str
    base_log_odds: float
    model_log_odds: float
    factors: list[Factor]
    units: str
    note: str


class RiskScoreDetail(BaseModel):
    provenance: Literal["simulated_prediction"] = "simulated_prediction"
    id: int
    scoring_run_id: int
    episode_id: str
    patient_id: str
    model_version: str
    model_type: str
    probability: float
    risk_tier: Tier
    rank: int
    model_log_odds: float
    prediction_at: datetime
    scored_at: datetime
    explanation: Explanation
    features: list[FeatureValue]
    run: ScoringRunOut
    is_latest_run: bool
    score_context: ScoreContext
    interpretation: str


class RiskScoreHistoryItem(BaseModel):
    provenance: Literal["simulated_prediction"] = "simulated_prediction"
    id: int
    scoring_run_id: int
    model_version: str
    probability: float
    risk_tier: Tier
    rank: int
    prediction_at: datetime
    scored_at: datetime


# ---------------------------------------------------------------- outreach
class OutreachCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel: Channel
    outcome: Outcome
    attempted_at: AwareDatetime | None = Field(
        default=None, description="When the contact attempt happened (default: now). Must not be in "
                                  "the future.")
    note: str | None = Field(default=None, max_length=2000)
    next_action: NextAction = "none"
    next_action_due: date | None = None

    @field_validator("note")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        v = v.strip() if v else v
        return v or None

    @model_validator(mode="after")
    def _rules(self) -> OutreachCreate:
        now = datetime.now(UTC)
        if self.attempted_at is not None and self.attempted_at > now + timedelta(minutes=5):
            raise ValueError("attempted_at must not be in the future")
        attempted = (self.attempted_at or now).astimezone(UTC).date()
        if self.next_action == "none" and self.next_action_due is not None:
            raise ValueError("next_action_due must be empty when next_action is 'none'")
        if self.next_action != "none" and self.next_action_due is None:
            raise ValueError("next_action_due is required when a next action is set")
        if self.next_action_due is not None and self.next_action_due < attempted:
            raise ValueError("next_action_due must not be before the attempt date")
        return self


class OutreachOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    provenance: Literal["recorded_outreach_outcome"] = "recorded_outreach_outcome"
    id: str
    patient_id: str
    episode_id: str | None
    programme: str
    origin: str
    purpose: str
    channel: str
    outcome: str
    note: str | None
    next_action: str
    next_action_due: date | None
    attempted_at: datetime
    recorded_at: datetime
    recorded_by: str


class OutreachListItem(OutreachOut):
    patient: PatientSummary
    follow_up_status: FollowUpStatus | None = None


class OutcomesSummary(BaseModel):
    as_of: date
    attempts_by_month: list[dict[str, Any]] = Field(default_factory=list)
    attempts_by_origin: dict[str, int]
    attempts_by_outcome: dict[str, dict[str, int]]
    attempts_by_channel: dict[str, int]
    phone_contact_rate: dict[str, float | None]
    app_open_next_actions: int
    completion_by_year: list[dict[str, Any]]
    completion_by_programme: list[dict[str, Any]]
    completion_by_outreach_exposure: list[dict[str, Any]]
    notes: list[str]


# ---------------------------------------------------------------- model / data quality / audit
class ModelInfo(BaseModel):
    provenance: Literal["simulated_prediction"] = "simulated_prediction"
    model_version: str
    model_type: str
    calibration_method: str
    feature_spec_version: str
    tier_thresholds: dict[str, float]
    data_fingerprint: str
    prediction_target: str
    trained_on: dict[str, Any]
    calibrated_on: dict[str, Any]
    selection: dict[str, Any]
    splits: dict[str, Any]
    test_metrics: dict[str, dict[str, Any]]
    test_intervals: dict[str, dict[str, list[float]]]
    validation_metrics: dict[str, dict[str, Any]]
    library_versions: dict[str, str]
    calibration_table: list[dict[str, float]] = Field(default_factory=list)
    test_tiers: list[dict[str, Any]] = Field(default_factory=list)
    feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    synthetic_data_only: bool
    disclaimer: str


class ModelDocuments(BaseModel):
    model_version: str
    model_card_markdown: str | None
    model_report_markdown: str | None


class AppUserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    display_name: str | None
    role: str
    is_active: bool
    linked_to_auth_identity: bool
    created_at: datetime
    last_sign_in_at: datetime | None


class DataQuality(BaseModel):
    status: Literal["ok", "warning"]
    checks: list[dict[str, Any]]
    dataset: dict[str, Any] | None
    table_counts: dict[str, dict[str, int | None]]
    label_status_counts: dict[str, int]
    feature_missingness: dict[str, float]
    latest_scoring_run: ScoringRunOut | None


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    occurred_at: datetime
    actor_id: str
    actor_role: str
    action: str
    entity_type: str
    entity_id: str | None
    details: dict[str, Any]
