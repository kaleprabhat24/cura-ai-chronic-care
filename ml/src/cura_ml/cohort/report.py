"""Programmatic cohort summary report (Markdown).

Every number in the report is computed here from the persisted dataset. Prediction-time availability
statistics only use records with recorded_at strictly before each episode's prediction timestamp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import RECORD_TABLES, load_records, read_table
from cura_ml.manifest import read_manifest
from cura_ml.synth import catalog as cat


def _table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, bool | np.bool_):
        return str(bool(v))
    if isinstance(v, float | np.floating):
        return f"{v:,.3f}" if abs(v) < 1 else f"{v:,.1f}"
    if isinstance(v, int | np.integer):
        return f"{int(v):,}"
    return str(v)


def _counts(series: pd.Series, name: str) -> pd.DataFrame:
    vc = series.value_counts(dropna=False)
    out = pd.DataFrame({name: [str(i) for i in vc.index], "count": [int(v) for v in vc.to_numpy()]})
    out["share"] = [round(c / max(len(series), 1), 3) for c in out["count"]]
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _last_before(episodes: pd.DataFrame, events: pd.DataFrame, time_col: str) -> pd.Series:
    """For each episode, the latest `time_col` of the patient's events strictly before t0."""
    left = episodes[["episode_id", "patient_id", "prediction_at"]].sort_values("prediction_at")
    right = pd.DataFrame({"patient_id": events["patient_id"].astype(str),
                          "event_time": as_utc(events[time_col])}).sort_values("event_time")
    merged = pd.merge_asof(left, right, left_on="prediction_at", right_on="event_time",
                           by="patient_id", allow_exact_matches=False, direction="backward")
    return merged.set_index("episode_id")["event_time"].reindex(episodes["episode_id"])


def build_report(data_dir: str | Path, cfg: dict[str, Any]) -> str:
    rec = load_records(data_dir)
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    labels = read_table(data_dir, "outcomes", "episode_labels")
    manifest = read_manifest(data_dir)
    episodes["prediction_at"] = as_utc(episodes["prediction_at"])
    cutoff = pd.Timestamp(cfg["simulation"]["data_cutoff"], tz="UTC")
    ep = episodes.merge(labels, on="episode_id", validate="one_to_one")
    elig = ep[ep["eligible"]].copy()
    lab = ep[ep["label_status"] == "labelled"].copy()
    band = cfg["checks"]["label_prevalence_band"]
    out: list[str] = []
    w = out.append

    w(f"# Cura AI synthetic cohort summary — {cfg['dataset_version']}\n")
    w("> Generated programmatically by `cura-ml report` from the persisted dataset. **All data are "
      "synthetic.** Nothing here is evidence of real-world clinical behaviour or model performance.\n")

    w("## Provenance\n")
    w(_table(pd.DataFrame([
        ("dataset_version", cfg["dataset_version"]),
        ("generator_version", manifest["generator_version"]),
        ("seed", str(cfg["seed"])),
        ("config_sha256", manifest["config_sha256"][:16] + "…"),
        ("history start", cfg["simulation"]["start_date"]),
        ("data cutoff (simulated today)", cfg["simulation"]["data_cutoff"]),
        ("python / numpy / pandas / pyarrow", " / ".join(
            manifest["library_versions"][k] for k in ("python", "numpy", "pandas", "pyarrow"))),
    ], columns=["item", "value"])) + "\n")

    w("## Record tables\n")
    w(_table(pd.DataFrame([(f"records/{t}", len(getattr(rec, t))) for t in RECORD_TABLES]
                          + [(k, v["rows"]) for k, v in manifest["files"].items()
                             if not k.startswith("records/")],
                          columns=["table", "rows"])) + "\n")

    pts = rec.patients
    age = (cutoff - as_utc(pts["birth_date"])).dt.days / 365.25
    n_progs = rec.conditions.groupby("patient_id")["code"].nunique().reindex(pts["patient_id"])
    w("## Patients\n")
    w(_table(pd.DataFrame([
        ("patients", len(pts)),
        ("age at cutoff, median (IQR)",
         f"{age.median():.1f} ({age.quantile(.25):.1f}–{age.quantile(.75):.1f})"),
        ("female share", _pct((pts["sex"] == "female").mean())),
        ("registered after history start", _pct((as_utc(pts["registered_on"])
                                                 > pd.Timestamp(cfg["simulation"]["start_date"],
                                                                tz="UTC")).mean())),
        ("deaths recorded by cutoff", int(pts["death_date"].notna().sum())),
        ("transfers out recorded by cutoff", int(pts["transfer_out_date"].notna().sum())),
        ("with type 2 diabetes recorded",
         int(rec.conditions.loc[rec.conditions["code"] == "44054006", "patient_id"].nunique())),
        ("with hypertension recorded",
         int(rec.conditions.loc[rec.conditions["code"] == "38341003", "patient_id"].nunique())),
        ("with both conditions", int((n_progs == 2).sum())),
        ("with no programme condition recorded", int(n_progs.isna().sum())),
    ], columns=["measure", "value"])) + "\n")
    w(_table(_counts(pts["deprivation_quintile"].sort_values(), "deprivation_quintile")) + "\n")

    w("## Care-process records\n")
    w("### Appointments by status\n")
    w(_table(_counts(rec.appointments["status"], "status")) + "\n")
    w("### Appointments by booking source\n")
    w(_table(_counts(rec.appointments["booking_source"], "booking_source")) + "\n")
    w("### Encounters by type\n")
    w(_table(_counts(rec.encounters["encounter_type"], "encounter_type")) + "\n")
    w("### Observations by code\n")
    obs = rec.observations.groupby(["code", "display", "unit"], as_index=False).agg(
        count=("value_num", "size"), median_value=("value_num", "median"))
    w(_table(obs) + "\n")
    w("### Medication statements\n")
    meds = rec.medication_statements
    med = meds.groupby("display", as_index=False).agg(
        statements=("medication_statement_id", "size"),
        ended_by_cutoff=("end_date", lambda s: int(s.notna().sum())))
    w(_table(med) + "\n")
    w("### Historical outreach attempts\n")
    orr = rec.outreach_attempts.groupby(["purpose", "outcome"], as_index=False).size()
    w(_table(orr.rename(columns={"size": "count"})) + "\n")

    w("## Follow-up episodes\n")
    w(f"Episodes are reconstructed from ordinary records only. Prediction timestamp "
      f"t0 = due date − {cfg['episodes']['prediction_lead_days']} days; label window "
      f"[t0, due + {cfg['episodes']['completion_grace_days']} days]; outcome mature when cutoff ≥ "
      f"due + {cfg['episodes']['completion_grace_days'] + cfg['episodes']['recording_lag_days']} "
      f"days.\n")
    w(_table(pd.DataFrame([
        ("episodes (all)", len(ep)),
        ("eligible at t0", len(elig)),
        ("eligible, cold start (< "
         f"{cfg['episodes']['cold_start_history_days']} days history)",
         int(elig["is_cold_start"].fillna(False).sum())),
        ("eligible, labelled (mature, uncensored)", len(lab)),
        ("eligible, censored", int((ep["label_status"] == "censored").sum())),
        ("eligible, immature at cutoff", int((ep["label_status"] == "immature").sum())),
        ("eligible, no completion recorded by cutoff (any year)",
         int(elig["completed_at"].isna().sum())),
        ("currently open at cutoff (eligible, immature, not yet completed) — worklist candidates",
         int(((elig["label_status"] == "immature") & elig["completed_at"].isna()).sum())),
    ], columns=["measure", "value"])) + "\n")
    w("### Ineligibility reasons\n")
    w(_table(_counts(ep.loc[~ep["eligible"], "ineligible_reason"], "reason")) + "\n")
    w("### Label status by programme\n")
    w(_table(pd.crosstab(ep["programme"], ep["label_status"]).reset_index()) + "\n")

    w("## Label prevalence (labelled episodes)\n")
    prev = float(lab["label"].astype(float).mean()) if len(lab) else float("nan")
    in_band = band[0] <= prev <= band[1]
    w(f"Overall prevalence of *follow-up incomplete at due + "
      f"{cfg['episodes']['completion_grace_days']} days*: **{_pct(prev)}** "
      f"(n = {len(lab):,}). Configured plausibility band {_pct(band[0])}–{_pct(band[1])}: "
      f"**{'PASS' if in_band else 'FAIL'}**.\n")
    lab["prediction_year"] = as_utc(lab["prediction_at"]).dt.year.astype(str)
    for key in ("programme", "prediction_year"):
        g = lab.groupby(key)["label"].agg(["size", "mean"]).reset_index()
        g.columns = [key, "labelled", "prevalence"]
        g["prevalence"] = g["prevalence"].astype(float).round(3)
        w(_table(g) + "\n")

    w("## Outcome maturity (eligible episodes by prediction year)\n")
    elig["prediction_year"] = elig["prediction_at"].dt.year.astype(str)
    w(_table(pd.crosstab(elig["prediction_year"], elig["label_status"]).reset_index()) + "\n")

    w("## Missingness\n")
    w("### Null share in record columns (columns with any nulls)\n")
    rows = []
    for t in RECORD_TABLES:
        df = getattr(rec, t)
        for c in df.columns:
            share = float(df[c].isna().mean()) if len(df) else 0.0
            if share > 0:
                rows.append((t, c, round(share, 3)))
    w(_table(pd.DataFrame(rows, columns=["table", "column", "null_share"])) + "\n")
    w("Most nulls are structural (e.g. `death_date` for living patients, `appointment_id` for "
      "walk-in acute encounters, `end_date` for ongoing medicines).\n")

    w("### Information available at prediction time (eligible episodes)\n")
    w("Computed only from records with `recorded_at` (or `status_recorded_at` / `booked_at`) "
      "strictly before t0.\n")
    avail = []
    year = pd.Timedelta(days=365)
    t2d = elig[elig["programme"] == "t2d"]
    for label_txt, subset, events, col in (
        ("T2D episodes: HbA1c recorded in prior 365 days", t2d,
         rec.observations[rec.observations["code"] == cat.OBSERVATIONS["hba1c"][0]], "recorded_at"),
        ("All episodes: systolic BP recorded in prior 365 days", elig,
         rec.observations[rec.observations["code"] == cat.OBSERVATIONS["sbp"][0]], "recorded_at"),
        ("All episodes: no-show recorded in prior 365 days", elig,
         rec.appointments[rec.appointments["status"] == "no_show"], "status_recorded_at"),
        ("All episodes: outreach attempt recorded in prior 365 days", elig,
         rec.outreach_attempts, "recorded_at"),
    ):
        last = _last_before(subset, events, col)
        share = ((subset["prediction_at"].to_numpy() - last.to_numpy())
                 <= year.to_timedelta64()).mean() if len(subset) else float("nan")
        avail.append((label_txt, _pct(float(share))))
    # Forward booking known at t0: booked before t0, scheduled in [t0, due], not cancelled before t0.
    ap = rec.appointments.assign(booked_at=as_utc(rec.appointments["booked_at"]),
                                 scheduled_for=as_utc(rec.appointments["scheduled_for"]),
                                 status_recorded_at=as_utc(rec.appointments["status_recorded_at"]))
    enc_type = {n: p["encounter_type"] for n, p in cfg["programmes"].items()}
    e = elig.assign(appointment_type=elig["programme"].map(enc_type),
                    due_ts=as_utc(elig["due_date"]))
    m = e[["episode_id", "patient_id", "appointment_type", "prediction_at", "due_ts"]].merge(
        ap[["patient_id", "appointment_type", "booked_at", "scheduled_for", "status",
            "status_recorded_at"]], on=["patient_id", "appointment_type"])
    cancelled_before = m["status"].str.startswith("cancelled") & \
        (m["status_recorded_at"] < m["prediction_at"])
    fwd = m[(m["booked_at"] < m["prediction_at"]) & (m["scheduled_for"] >= m["prediction_at"])
            & (m["scheduled_for"] < m["due_ts"] + pd.Timedelta(days=1)) & ~cancelled_before]
    avail.append(("All episodes: review appointment already booked for [t0, due] at t0",
                  _pct(fwd["episode_id"].nunique() / max(len(elig), 1))))
    w(_table(pd.DataFrame(avail, columns=["measure", "share of eligible episodes"])) + "\n")

    w("## Separation of information classes\n")
    w("- `records/` — ordinary records as known at the cutoff (future model inputs).\n"
      "- `cohort/` — episode definitions and eligibility, known at t0.\n"
      "- `outcomes/` — completion, censoring, maturity and labels (targets only, never features).\n"
      "- `hidden/` — latent simulation traits and the simulator's episode log (validation only).\n")
    return "\n".join(out)


def write_report(data_dir: str | Path, cfg: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(data_dir, cfg), encoding="utf-8")
    return path
