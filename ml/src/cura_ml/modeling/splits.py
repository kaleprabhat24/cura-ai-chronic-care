"""Patient-level and time-aware evaluation splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cura_ml.cohort.episodes import as_utc

SPLITS = ("train", "val_tune", "val_cal", "test")


def assign_splits(episodes: pd.DataFrame, labels: pd.DataFrame, cohort_cfg: dict[str, Any],
                  train_cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per labelled episode with its split (`train`, `val_tune`, `val_cal`, `test` or
    `unused`) and label. Patients are assigned to the hold-out group and to the validation halves
    with a seeded generator, so splits are reproducible."""
    sc = train_cfg["splits"]
    rng = np.random.default_rng(train_cfg["seed"])
    df = episodes[["episode_id", "patient_id", "programme", "prediction_at", "window_end_date"]].merge(
        labels[["episode_id", "label_status", "label"]], on="episode_id", validate="one_to_one")
    df = df[df["label_status"] == "labelled"].copy()
    df["year"] = as_utc(df["prediction_at"]).dt.year
    lag = pd.Timedelta(days=cohort_cfg["episodes"]["recording_lag_days"])
    df["outcome_known_at"] = as_utc(df["window_end_date"]) + lag

    patients = np.array(sorted(episodes["patient_id"].unique()))
    holdout = set(rng.choice(patients, size=int(round(sc["holdout_patient_fraction"] * len(patients))),
                             replace=False))
    rest = np.array([p for p in patients if p not in holdout])
    cal = set(rng.choice(rest, size=int(round(sc["validation_calibration_fraction"] * len(rest))),
                         replace=False))

    val_start = pd.Timestamp(f"{sc['validation_year']}-01-01", tz="UTC")
    test_start = pd.Timestamp(f"{sc['test_year']}-01-01", tz="UTC")
    is_hold = df["patient_id"].isin(holdout)
    split = np.full(len(df), "unused", dtype=object)
    train = (~is_hold & df["year"].isin(sc["train_years"]) & (df["outcome_known_at"] < val_start))
    val = (~is_hold & (df["year"] == sc["validation_year"]) & (df["outcome_known_at"] < test_start))
    test = is_hold & (df["year"] == sc["test_year"])
    split[train.to_numpy()] = "train"
    split[(val & ~df["patient_id"].isin(cal)).to_numpy()] = "val_tune"
    split[(val & df["patient_id"].isin(cal)).to_numpy()] = "val_cal"
    split[test.to_numpy()] = "test"
    df["split"] = split
    df["holdout_patient"] = is_hold
    df["label"] = df["label"].astype(int)
    return df[["episode_id", "patient_id", "programme", "year", "split", "holdout_patient",
               "outcome_known_at", "label"]].reset_index(drop=True)
