"""Reconstruct follow-up episodes and target labels from ordinary records only.

Implements docs/PREDICTION_TARGET.md. Inputs are a `Records` snapshot (as known at the data cutoff)
and the config; hidden simulation traits are never read here.

Output is split in two tables on purpose:
  * followup_episodes  — what is known at the prediction timestamp t0 (definition + eligibility)
  * episode_labels     — what happens after t0 (completion, censoring, maturity, label)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from cura_ml.dataset import Records

UTC_NS = "datetime64[ns, UTC]"
DAY = pd.Timedelta(days=1)

EPISODE_COLUMNS = [
    "episode_id", "patient_id", "programme", "condition_code", "index_event_at",
    "index_event_type", "index_event_ref", "due_date", "prediction_at", "window_end_date",
    "eligible", "ineligible_reason", "is_cold_start", "history_days",
]
LABEL_COLUMNS = [
    "episode_id", "completed_at", "completion_event_type", "completion_event_ref",
    "completed_in_window", "censored", "censor_reason", "censor_date", "mature", "label_status",
    "label",
]
# Checked in this order; the first matching reason is reported.
INELIGIBLE_REASONS = (
    "prediction_after_cutoff",
    "deceased_before_prediction",
    "transferred_before_prediction",
    "condition_not_recorded",
    "no_prior_encounter",
    "completed_before_prediction",
)


def as_utc(values: pd.Series) -> pd.Series:
    """Convert dates/timestamps (any resolution, naive = UTC) to tz-aware UTC nanoseconds."""
    s = pd.to_datetime(values)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.astype(UTC_NS)


def _dates(ts: pd.Series) -> pd.Series:
    values = [None if pd.isna(x) else x.date() for x in ts]
    return pd.Series(pd.array(values, dtype=pd.ArrowDtype(pa.date32())), index=ts.index)


def qualifying_events(records: Records, cfg: dict[str, Any], known_before: pd.Timestamp) -> pd.DataFrame:
    """Events that count as completing a programme review, restricted to those recorded before
    `known_before`. Rank breaks timestamp ties: encounter < appointment < observation."""
    enc = records.encounters
    enc = enc[(enc["status"] == "finished") & (as_utc(enc["recorded_at"]) < known_before)]
    ap = records.appointments
    ap = ap[(ap["status"] == "attended") & (as_utc(ap["status_recorded_at"]) < known_before)]
    obs = records.observations
    obs = obs[as_utc(obs["recorded_at"]) < known_before]
    parts = []
    for name, p in cfg["programmes"].items():
        e = enc[enc["encounter_type"] == p["encounter_type"]]
        a = ap[ap["appointment_type"] == p["encounter_type"]]
        o = obs[obs["code"].isin(p["completion_observation_codes"])]
        for df, time_col, ref_col, kind, rank in (
            (e, "start_at", "encounter_id", "encounter", 0),
            (a, "scheduled_for", "appointment_id", "appointment", 1),
            (o, "effective_at", "observation_id", "observation", 2),
        ):
            parts.append(pd.DataFrame({
                "patient_id": df["patient_id"].astype(str).to_numpy(),
                "programme": name,
                "effective_at": as_utc(df[time_col]).to_numpy(),
                "event_type": kind,
                "event_ref": df[ref_col].astype(str).to_numpy(),
                "rank": rank,
            }))
    q = pd.concat(parts, ignore_index=True)
    q["effective_at"] = as_utc(q["effective_at"])
    return q.sort_values(["patient_id", "programme", "effective_at", "rank", "event_ref"],
                         ignore_index=True)


def _cluster(q: pd.DataFrame, merge_window: pd.Timedelta) -> pd.DataFrame:
    """Group qualifying events into reviews. A review starts at the first event that is at least
    `merge_window` after the previous review start. Each review start is the index event of one
    episode and the completion event of the episode before it."""
    rows = []
    for (pid, prog), g in q.groupby(["patient_id", "programme"], sort=True):
        starts: list[tuple] = []
        anchor = None
        for eff, kind, ref in zip(g["effective_at"], g["event_type"], g["event_ref"], strict=True):
            if anchor is None or eff >= anchor + merge_window:
                starts.append((eff, kind, ref))
                anchor = eff
        for i, (eff, kind, ref) in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else (pd.NaT, None, None)
            rows.append((pid, prog, eff, kind, ref, *nxt))
    return pd.DataFrame(rows, columns=[
        "patient_id", "programme", "index_event_at", "index_event_type", "index_event_ref",
        "completed_at", "completion_event_type", "completion_event_ref",
    ])


def build_episodes(records: Records, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ecfg = cfg["episodes"]
    cutoff = pd.Timestamp(cfg["simulation"]["data_cutoff"], tz="UTC").as_unit("ns")
    known_before = cutoff + DAY  # dataset = everything recorded by the end of the cutoff day

    ep = _cluster(qualifying_events(records, cfg, known_before),
                  pd.Timedelta(days=ecfg["merge_window_days"]))
    if ep.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS), pd.DataFrame(columns=LABEL_COLUMNS)
    for col in ("index_event_at", "completed_at"):
        ep[col] = as_utc(ep[col])

    progs = cfg["programmes"]
    interval = pd.to_timedelta(ep["programme"].map({n: p["interval_days"] for n, p in progs.items()}),
                               unit="D")
    ep["due_ts"] = due = ep["index_event_at"].dt.floor("D") + interval
    t0 = due - pd.Timedelta(days=ecfg["prediction_lead_days"])
    ep["window_end_ts"] = due + pd.Timedelta(days=ecfg["completion_grace_days"])
    ep["condition_code"] = ep["programme"].map({n: p["condition_code"] for n, p in progs.items()})
    ep["prediction_at"] = t0
    ep["episode_id"] = ("ep_" + ep["patient_id"].str.removeprefix("pt_") + "_" + ep["programme"]
                        + "_" + ep["index_event_at"].dt.strftime("%Y%m%d"))

    # Condition recorded (earliest record of the programme's condition code).
    cond = records.conditions.assign(recorded_at=as_utc(records.conditions["recorded_at"]))
    cond = cond[cond["recorded_at"] < known_before]
    first_cond = cond.groupby(["patient_id", "code"], as_index=False)["recorded_at"].min()
    first_cond = first_cond.rename(columns={"code": "condition_code", "recorded_at": "cond_rec"})
    ep = ep.merge(first_cond, on=["patient_id", "condition_code"], how="left")

    # Encounter history recorded strictly before t0 (earliest start among those).
    enc = records.encounters
    enc = pd.DataFrame({"patient_id": enc["patient_id"].astype(str),
                        "enc_rec": as_utc(enc["recorded_at"]),
                        "start_at": as_utc(enc["start_at"])})
    enc = enc[enc["enc_rec"] < known_before].sort_values(["patient_id", "enc_rec"])
    enc["first_start"] = enc.groupby("patient_id")["start_at"].cummin()
    hist = pd.merge_asof(
        ep[["episode_id", "patient_id", "prediction_at"]].sort_values("prediction_at"),
        enc[["patient_id", "enc_rec", "first_start"]].sort_values("enc_rec"),
        left_on="prediction_at", right_on="enc_rec", by="patient_id",
        allow_exact_matches=False, direction="backward",
    )
    ep = ep.merge(hist[["episode_id", "enc_rec", "first_start"]], on="episode_id", how="left")

    # Exits known at the cutoff.
    pts = records.patients
    exits = pd.DataFrame({"patient_id": pts["patient_id"].astype(str)})
    for day_col, rec_col, out in (("death_date", "death_recorded_at", "death_ts"),
                                  ("transfer_out_date", "transfer_recorded_at", "transfer_ts")):
        ts, rec = as_utc(pts[day_col]), as_utc(pts[rec_col])
        exits[out] = ts.where(rec.notna() & (rec < known_before))
    ep = ep.merge(exits, on="patient_id", how="left")

    t0 = ep["prediction_at"]
    conditions = [
        t0 > cutoff,
        ep["death_ts"].notna() & (ep["death_ts"] <= t0),
        ep["transfer_ts"].notna() & (ep["transfer_ts"] <= t0),
        ep["cond_rec"].isna() | (ep["cond_rec"] >= t0),
        ep["enc_rec"].isna(),
        ep["completed_at"].notna() & (ep["completed_at"] < t0),
    ]
    reason = np.select(conditions, INELIGIBLE_REASONS, default="")
    eligible = reason == ""
    history_days = (t0 - ep["first_start"]).dt.days

    episodes = pd.DataFrame({
        "episode_id": ep["episode_id"],
        "patient_id": ep["patient_id"],
        "programme": ep["programme"],
        "condition_code": ep["condition_code"],
        "index_event_at": ep["index_event_at"],
        "index_event_type": ep["index_event_type"],
        "index_event_ref": ep["index_event_ref"],
        "due_date": _dates(ep["due_ts"]),
        "prediction_at": t0,
        "window_end_date": _dates(ep["window_end_ts"]),
        "eligible": eligible,
        "ineligible_reason": pd.Series(np.where(eligible, None, reason), dtype=object),
        "is_cold_start": pd.array(
            np.where(history_days.isna(), None,
                     history_days < ecfg["cold_start_history_days"]), dtype="boolean"),
        "history_days": pd.array(history_days, dtype="Int64"),
    })

    # ---------------- outcomes (future relative to t0)
    window_end = ep["window_end_ts"]
    completed_in_window = (ep["completed_at"].notna()
                           & (ep["completed_at"] < window_end + DAY)).to_numpy()
    exit_ts = ep[["death_ts", "transfer_ts"]].min(axis=1)
    censored = eligible & ~completed_in_window & exit_ts.notna().to_numpy() & \
        (exit_ts <= window_end).to_numpy()
    mature = (cutoff >= window_end + pd.Timedelta(days=ecfg["recording_lag_days"])).to_numpy()
    status = np.select([~eligible, censored, ~mature], ["ineligible", "censored", "immature"],
                       default="labelled")
    label = np.where(status == "labelled", (~completed_in_window).astype(int), -1)
    censor_reason = np.where(~censored, None,
                             np.where(ep["death_ts"].notna(), "death", "transfer_out"))

    labels = pd.DataFrame({
        "episode_id": ep["episode_id"],
        "completed_at": ep["completed_at"],
        "completion_event_type": ep["completion_event_type"],
        "completion_event_ref": ep["completion_event_ref"],
        "completed_in_window": pd.array(np.where(eligible, completed_in_window, None),
                                        dtype="boolean"),
        "censored": censored,
        "censor_reason": pd.Series(censor_reason, dtype=object),
        "censor_date": _dates(exit_ts.where(censored)),
        "mature": mature,
        "label_status": status,
        "label": pd.array(np.where(label < 0, None, label), dtype="Int8"),
    })
    order = episodes.sort_values(["patient_id", "programme", "index_event_at"]).index
    return (episodes.loc[order].reset_index(drop=True)[EPISODE_COLUMNS],
            labels.loc[order].reset_index(drop=True)[LABEL_COLUMNS])
