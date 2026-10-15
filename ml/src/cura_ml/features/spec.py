"""Declarative feature specification (version fs-1.0).

Each feature declares its source, time window, aggregation, missing-data policy and the plain-language
label used in explanations. `build.py` implements exactly these definitions, and
`docs/FEATURES.md` is generated from this module (`cura-ml features`).

Every source event is filtered on its *recorded* time (strictly before t0) before any aggregation.
Windows are measured backwards from t0 using the event's effective time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FEATURE_SPEC_VERSION = "fs-1.0"

LOOKBACK_DAYS = 730          # observation window (plan: 24 months)
SHORT_WINDOW_DAYS = 365
TREND_MIN_SPAN_DAYS = 180    # a trend needs >= 2 values spanning at least this many days

# Fixed smoothing priors (constants, not fitted, so they cannot leak information from any split).
ATTENDANCE_PRIOR, ATTENDANCE_PRIOR_STRENGTH = 0.80, 2.0
COMPLETION_PRIOR, COMPLETION_PRIOR_STRENGTH = 0.70, 2.0
CONTACT_PRIOR, CONTACT_PRIOR_STRENGTH = 0.50, 2.0

NEVER_MISSING = "never missing (zero/empty history gives 0 or the smoothing prior)"


@dataclass(frozen=True)
class Feature:
    name: str
    label: str            # plain-language explanation label
    unit: str
    group: str
    source: str
    window: str
    aggregation: str
    missing_policy: str
    indicator: bool = False  # True -> builder adds `<name>__missing` (1 when value is missing)
    missing_label: str = "not recorded"  # how a missing value is described in explanations


def _missing(what: str) -> str:
    return (f"NaN when {what}; companion indicator `__missing`; median-imputed for logistic "
            f"regression (median fitted on training data only); native NaN handling in XGBoost")


FEATURES: tuple[Feature, ...] = (
    # ---------------------------------------------------------------- attendance
    Feature("prior_appointments_24m", "Review appointments with a known outcome (24 mo)", "count",
            "attendance", "appointments (review types, all programmes)", "730 d",
            "count of attended + no-show appointments whose outcome was recorded before t0", NEVER_MISSING),
    Feature("noshows_12m", "Missed review appointments (12 mo)", "count", "attendance",
            "appointments", "365 d", "count of no-shows recorded before t0", NEVER_MISSING),
    Feature("noshows_24m", "Missed review appointments (24 mo)", "count", "attendance",
            "appointments", "730 d", "count of no-shows recorded before t0", NEVER_MISSING),
    Feature("attendance_rate_24m_smoothed", "Attendance rate, smoothed (24 mo)", "ratio",
            "attendance", "appointments", "730 d",
            f"(attended + {ATTENDANCE_PRIOR_STRENGTH:g}×{ATTENDANCE_PRIOR:g}) / "
            f"(attended + no-shows + {ATTENDANCE_PRIOR_STRENGTH:g})", NEVER_MISSING),
    Feature("consecutive_noshows", "Consecutive most recent missed appointments", "count",
            "attendance", "appointments", "730 d",
            "no-shows after the last attended appointment (outcomes recorded before t0)",
            NEVER_MISSING),
    Feature("patient_cancellations_12m", "Appointments cancelled by the patient (12 mo)", "count",
            "attendance", "appointments", "365 d", "count of patient cancellations recorded before t0",
            NEVER_MISSING),
    Feature("days_since_last_visit", "Days since last attended visit", "days", "attendance",
            "encounters (any type)", "all history", "t0 − start of latest encounter recorded before t0",
            _missing("no encounter was recorded before t0"), indicator=True),
    Feature("acute_visits_12m", "Unplanned (acute) visits (12 mo)", "count", "attendance",
            "encounters (acute)", "365 d", "count", NEVER_MISSING),
    # ---------------------------------------------------------------- booking for this review
    Feature("has_open_booking", "Review appointment already booked for this follow-up", "yes/no",
            "booking", "appointments (this programme)", "booked before t0, slot ≥ t0",
            "1 if a booking made before t0 for a slot at/after t0 was still open at t0 "
            "(not cancelled/resolved before t0)", NEVER_MISSING),
    Feature("booking_lead_days", "Booking lead time of the open booking", "days", "booking",
            "appointments (this programme)", "open booking at t0", "slot − booking time (earliest open slot)",
            "NaN when there is no open booking (then has_open_booking = 0); median-imputed for "
            "logistic regression; native NaN in XGBoost", missing_label="no open booking"),
    Feature("booking_offset_from_due_days", "Booked slot relative to due date", "days", "booking",
            "appointments (this programme)", "open booking at t0", "slot date − due date",
            "NaN when there is no open booking; imputed as above", missing_label="no open booking"),
    # ---------------------------------------------------------------- prior follow-up outcomes
    Feature("prior_episodes_matured", "Earlier follow-ups with a known outcome", "count",
            "prior_followup", "follow-up episodes (this programme) + qualifying events", "all history",
            "earlier episodes whose window_end + recording lag ended before t0", NEVER_MISSING),
    Feature("prior_ontime_rate_smoothed", "Earlier follow-ups completed on time, smoothed", "ratio",
            "prior_followup", "earlier matured episodes; qualifying events recorded before t0",
            "all history", f"(on time + {COMPLETION_PRIOR_STRENGTH:g}×{COMPLETION_PRIOR:g}) / "
            f"(matured + {COMPLETION_PRIOR_STRENGTH:g})", NEVER_MISSING),
    Feature("last_prior_followup_late", "Most recent known follow-up was late", "yes/no",
            "prior_followup", "earlier matured episodes", "all history",
            "1 if the latest matured episode was not completed by due + 30 d",
            _missing("there is no matured earlier episode"), indicator=True,
            missing_label="no earlier follow-up with a known outcome"),
    # ---------------------------------------------------------------- measurements
    Feature("hba1c_last", "Latest HbA1c", "%", "measurements", "observations LOINC 4548-4",
            "730 d", "latest value recorded before t0", _missing("no HbA1c in the window"),
            indicator=True, missing_label="no HbA1c in the last 24 months"),
    Feature("days_since_hba1c", "Days since latest HbA1c", "days", "measurements",
            "observations LOINC 4548-4", "730 d", "t0 − effective time of latest value",
            _missing("no HbA1c in the window"), missing_label="no HbA1c in the last 24 months"),
    Feature("hba1c_count_24m", "HbA1c results (24 mo)", "count", "measurements",
            "observations LOINC 4548-4", "730 d", "count", NEVER_MISSING),
    Feature("hba1c_trend_per_year", "HbA1c trend", "% per year", "measurements",
            "observations LOINC 4548-4", "730 d",
            f"least-squares slope; needs ≥ 2 values spanning ≥ {TREND_MIN_SPAN_DAYS} d",
            _missing("history is inadequate for a trend"), indicator=True,
            missing_label="too few values for a trend"),
    Feature("sbp_last", "Latest systolic blood pressure", "mm[Hg]", "measurements",
            "observations LOINC 8480-6", "730 d", "latest value recorded before t0",
            _missing("no systolic BP in the window"), indicator=True),
    Feature("days_since_sbp", "Days since latest blood pressure", "days", "measurements",
            "observations LOINC 8480-6", "730 d", "t0 − effective time of latest value",
            _missing("no systolic BP in the window")),
    Feature("sbp_count_24m", "Blood pressure readings (24 mo)", "count", "measurements",
            "observations LOINC 8480-6", "730 d", "count", NEVER_MISSING),
    Feature("sbp_trend_per_year", "Systolic blood pressure trend", "mm[Hg] per year",
            "measurements", "observations LOINC 8480-6", "730 d",
            f"least-squares slope; needs ≥ 2 values spanning ≥ {TREND_MIN_SPAN_DAYS} d",
            _missing("history is inadequate for a trend"), indicator=True,
            missing_label="too few values for a trend"),
    # ---------------------------------------------------------------- medications
    Feature("active_medications", "Active recorded medicines", "count", "medications",
            "medication_statements", "as of t0",
            "started (recorded) before t0 and not stopped as known at t0", NEVER_MISSING),
    Feature("programme_medications", "Active medicines for this condition", "count",
            "medications", "medication_statements", "as of t0",
            "active medicines on this programme's treatment ladder", NEVER_MISSING),
    Feature("medication_starts_12m", "Medicines started (12 mo)", "count", "medications",
            "medication_statements", "365 d", "starts recorded before t0", NEVER_MISSING),
    Feature("medication_stops_12m", "Medicines stopped (12 mo)", "count", "medications",
            "medication_statements", "365 d", "stops recorded before t0", NEVER_MISSING),
    # ---------------------------------------------------------------- outreach history
    Feature("outreach_calls_12m", "Outreach phone calls (12 mo)", "count", "outreach",
            "outreach_attempts (phone)", "365 d", "count recorded before t0", NEVER_MISSING),
    Feature("outreach_reached_12m", "Outreach calls that reached the patient (12 mo)", "count",
            "outreach", "outreach_attempts (phone)", "365 d", "count of reached_* outcomes",
            NEVER_MISSING),
    Feature("outreach_contact_rate_24m_smoothed", "Outreach contact rate, smoothed (24 mo)",
            "ratio", "outreach", "outreach_attempts (phone)", "730 d",
            f"(reached + {CONTACT_PRIOR_STRENGTH:g}×{CONTACT_PRIOR:g}) / (calls + "
            f"{CONTACT_PRIOR_STRENGTH:g})", NEVER_MISSING),
    Feature("recall_letters_12m", "Recall letters sent (12 mo)", "count", "outreach",
            "outreach_attempts (letter)", "365 d", "count recorded before t0", NEVER_MISSING),
    Feature("wrong_number_24m", "Wrong phone number recorded (24 mo)", "yes/no", "outreach",
            "outreach_attempts", "730 d", "1 if any wrong_number outcome", NEVER_MISSING),
    Feature("days_since_last_outreach", "Days since last outreach attempt", "days", "outreach",
            "outreach_attempts", "730 d", "t0 − latest attempt",
            _missing("no outreach in the window"), indicator=True,
            missing_label="no outreach in the last 24 months"),
    # ---------------------------------------------------------------- context / history length
    Feature("history_days", "Length of recorded encounter history", "days", "context",
            "encounters", "all history", "t0 − first encounter recorded before t0",
            _missing("no encounter was recorded before t0")),
    Feature("age_years", "Age at prediction time", "years", "context", "patients", "as of t0",
            "(t0 − birth date) / 365.25", NEVER_MISSING),
    Feature("chronic_conditions", "Chronic programme conditions recorded", "count", "context",
            "conditions", "as of t0", "distinct programme condition codes recorded before t0",
            NEVER_MISSING),
    Feature("programme_is_htn", "Follow-up is a hypertension review", "yes/no", "context",
            "followup_episodes", "episode definition", "1 for htn, 0 for t2d", NEVER_MISSING),
    Feature("deprivation_quintile", "Area deprivation quintile (5 = most deprived)", "quintile",
            "context", "patients", "as of t0", "recorded value", NEVER_MISSING),
)

FEATURE_BY_NAME = {f.name: f for f in FEATURES}


def indicator_name(name: str) -> str:
    return f"{name}__missing"


def feature_columns() -> list[str]:
    """Model input columns in a fixed order: each feature, followed by its missing indicator."""
    cols: list[str] = []
    for f in FEATURES:
        cols.append(f.name)
        if f.indicator:
            cols.append(indicator_name(f.name))
    return cols


def base_feature(column: str) -> str:
    return column.removesuffix("__missing")


def column_label(column: str) -> str:
    f = FEATURE_BY_NAME[base_feature(column)]
    return f"{f.label} — not recorded" if column.endswith("__missing") else f.label


def spec_records() -> list[dict]:
    return [asdict(f) for f in FEATURES]


def spec_markdown() -> str:
    lines = [
        f"# Feature specification ({FEATURE_SPEC_VERSION})", "",
        "Generated from `ml/src/cura_ml/features/spec.py` by `cura-ml features`. Do not edit by hand.",
        "",
        "All features are computed at the prediction timestamp t0 (due date − 14 days) from records "
        "whose **recorded** time is strictly before t0. Windows count back from t0 on effective time. "
        f"Observation window: {LOOKBACK_DAYS} days unless stated. Hidden simulation traits and episode "
        "outcomes are never inputs.", "",
        "| Feature | Explanation label | Unit | Source | Window | Aggregation | Missing-data policy |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in FEATURES:
        lines.append(f"| `{f.name}` | {f.label} | {f.unit} | {f.source} | {f.window} | "
                     f"{f.aggregation} | {f.missing_policy} |")
    lines += ["", f"Columns in the model matrix ({len(feature_columns())}): every feature above plus "
              "`<name>__missing` indicators for features marked with a companion indicator."]
    return "\n".join(lines) + "\n"
