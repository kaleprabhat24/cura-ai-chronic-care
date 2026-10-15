"""Load and validate cohort configuration files.

The config is kept as plain nested dicts (it maps 1:1 onto the YAML file), but every section and key
is validated against REQUIRED_KEYS so that a typo fails loudly instead of silently using nothing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS: dict[str, set[str]] = {
    "simulation": {"start_date", "data_cutoff", "n_patients"},
    "population": {
        "age_min", "age_max", "sex_probs", "language_probs", "registered_before_start_prob",
        "late_join_max_days", "programme_mix", "incident_condition_prob",
        "annual_death_hazard_at_60", "death_hazard_age_slope", "annual_transfer_out_rate",
    },
    "traits": {
        "access_deprivation_effect", "instability_entry_prob_range", "instability_exit_prob",
        "trust_engagement_corr",
    },
    "episodes": {
        "prediction_lead_days", "completion_grace_days", "recording_lag_days",
        "merge_window_days", "cold_start_history_days", "first_prediction_date",
    },
    "care_process": {
        "invite_lead_days", "prebook_base_logit", "prebook_engagement_weight",
        "prebook_offset_days", "invite_response_base_logit", "invite_response_engagement_weight",
        "invite_response_access_weight", "invite_response_unstable_weight",
        "booking_delay_mean_days", "booking_lead_days", "outreach_booking_lead_days",
        "clinic_rebook_lead_days", "clinic_cancel_prob", "patient_cancel_base_logit",
        "patient_cancel_unstable_weight", "rebook_after_cancel_base_logit",
        "rebook_after_noshow_base_logit", "rebook_engagement_weight", "rebook_delay_mean_days",
        "re_invite_every_days",
    },
    "attendance": {
        "base_logit", "engagement_weight", "access_weight", "unstable_weight", "long_lead_weight",
        "long_lead_cap_days",
        "recent_noshow_weight", "year_drift", "weekday_effects", "noise_sd",
    },
    "outreach_policy": {
        "pre_due_reminder_offset_days", "pre_due_reminder_prob", "pre_due_reminder_lookback_days",
        "chase_offsets_days", "chase_probs", "reach_base_logit", "reach_trust_weight",
        "reach_unstable_weight", "book_when_reached_base_logit",
        "book_when_reached_engagement_weight", "letter_response_base_logit",
        "letter_response_engagement_weight",
    },
    "clinical": {
        "hba1c_baseline", "hba1c_trend_per_year", "hba1c_trend_engagement_weight",
        "hba1c_drug_effect", "hba1c_measure_sd", "hba1c_intensify_threshold", "hba1c_not_done_base_logit",
        "hba1c_not_done_engagement_weight", "hba1c_not_done_unstable_weight", "hba1c_lab_failure_prob",
        "sbp_baseline",
        "sbp_trend_per_year", "sbp_trend_engagement_weight", "sbp_drug_effect", "sbp_measure_sd",
        "sbp_intensify_threshold", "intensify_prob", "med_stop_prob", "statin_prob_t2d",
        "bp_at_t2d_review_prob", "weight_baseline", "acute_visits_per_year",
        "acute_unstable_multiplier", "acute_bp_prob", "acute_weight_prob",
    },
    "checks": {"label_prevalence_band"},
}
TOP_LEVEL_SCALARS = {"dataset_version", "seed"}
PROGRAMME_KEYS = {"condition_code", "encounter_type", "interval_days", "completion_observation_codes"}


class ConfigError(ValueError):
    """Raised when a cohort config is missing keys or contains unknown keys."""


def validate_config(cfg: dict[str, Any]) -> None:
    expected_top = TOP_LEVEL_SCALARS | set(REQUIRED_KEYS) | {"programmes"}
    _check_keys("top level", set(cfg), expected_top)
    for section, keys in REQUIRED_KEYS.items():
        if not isinstance(cfg[section], dict):
            raise ConfigError(f"section '{section}' must be a mapping")
        _check_keys(f"section '{section}'", set(cfg[section]), keys)
    if not cfg["programmes"]:
        raise ConfigError("at least one programme is required")
    for name, prog in cfg["programmes"].items():
        _check_keys(f"programme '{name}'", set(prog), PROGRAMME_KEYS)
    if set(cfg["population"]["programme_mix"]) != {"t2d_only", "htn_only", "both"}:
        raise ConfigError("programme_mix must define t2d_only, htn_only and both")
    if set(cfg["programmes"]) != {"t2d", "htn"}:
        raise ConfigError("the v1 generator simulates exactly the 't2d' and 'htn' programmes")
    for probs_key in ("sex_probs", "language_probs", "programme_mix"):
        total = sum(cfg["population"][probs_key].values())
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"population.{probs_key} must sum to 1 (got {total})")
    if cfg["simulation"]["data_cutoff"] <= cfg["simulation"]["start_date"]:
        raise ConfigError("data_cutoff must be after start_date")
    if not (cfg["simulation"]["start_date"] <= cfg["episodes"]["first_prediction_date"]
            <= cfg["simulation"]["data_cutoff"]):
        raise ConfigError("episodes.first_prediction_date must lie within the simulated history")


def _check_keys(where: str, actual: set[str], expected: set[str]) -> None:
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        raise ConfigError(f"{where}: missing={sorted(missing)} unknown={sorted(unknown)}")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config, apply optional nested overrides (e.g. for tests) and validate it."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # YAML may parse quoted dates as strings and unquoted ones as dates; normalise to ISO strings.
    for key in ("start_date", "data_cutoff"):
        cfg["simulation"][key] = str(cfg["simulation"][key])
    cfg["episodes"]["first_prediction_date"] = str(cfg["episodes"]["first_prediction_date"])
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of the config (key order and whitespace independent)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
