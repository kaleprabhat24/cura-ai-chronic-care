"""Feature builder: one row per episode, computed as of the episode's prediction timestamp t0.

Leakage rules (enforced here and tested in ml/tests/test_features.py):
  * every source table is joined to episodes and filtered on its *recorded* timestamp strictly
    before t0 (`_known`) before any aggregation;
  * appointment statuses and medication stops are taken as they were known at t0;
  * earlier follow-up outcomes count only when matured before t0, and only from completion events
    recorded before t0;
  * hidden simulation traits and the outcomes tables are never inputs (this module cannot read them).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cura_ml.cohort.episodes import as_utc, qualifying_events
from cura_ml.dataset import Records
from cura_ml.features import spec as S
from cura_ml.synth import catalog as cat

DAY = pd.Timedelta(days=1)
FAR_FUTURE = pd.Timestamp("2262-01-01", tz="UTC")
OUTCOME_STATUSES = ("attended", "no_show")
REACHED = ("reached_booked", "reached_declined")


def records_as_of(records: Records, t: pd.Timestamp) -> Records:
    """The records as they were known strictly before time `t` (same semantics as the cutoff)."""
    def before(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[as_utc(df[col]) < t].copy()

    ap = before(records.appointments, "booked_at")
    pending = as_utc(ap["status_recorded_at"]) >= t
    ap.loc[pending, "status"] = "booked"
    ap.loc[pending, "status_recorded_at"] = ap.loc[pending, "booked_at"]
    ms = before(records.medication_statements, "recorded_at")
    unknown_end = ms["end_recorded_at"].notna() & (as_utc(ms["end_recorded_at"]) >= t)
    ms.loc[unknown_end, ["end_date", "end_recorded_at"]] = None
    pts = records.patients.copy()
    for day_col, rec_col in (("death_date", "death_recorded_at"),
                             ("transfer_out_date", "transfer_recorded_at")):
        unknown = pts[rec_col].notna() & (as_utc(pts[rec_col]) >= t)
        pts.loc[unknown, [day_col, rec_col]] = None
    return Records(
        patients=pts,
        conditions=before(records.conditions, "recorded_at"),
        appointments=ap,
        encounters=before(records.encounters, "recorded_at"),
        observations=before(records.observations, "recorded_at"),
        medication_statements=ms,
        outreach_attempts=before(records.outreach_attempts, "recorded_at"),
    )


def _utc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = as_utc(df[c])
    return df


def _known(ep: pd.DataFrame, events: pd.DataFrame, recorded_col: str, time_col: str | None = None,
           window_days: int | None = None, on: tuple[str, ...] = ("patient_id",)) -> pd.DataFrame:
    """Episode × event pairs for the same patient, keeping only events recorded strictly before t0
    and (optionally) with `time_col` inside the `window_days` before t0. This is the only way source
    events reach the aggregations below."""
    m = ep[["row", "t0", *on]].merge(events, on=list(on), how="inner")
    m = m[m[recorded_col] < m["t0"]]
    if window_days is not None:
        m = m[m[time_col] >= m["t0"] - pd.Timedelta(days=window_days)]
    return m


def _per_row(series: pd.Series, n: int, fill: float = np.nan) -> np.ndarray:
    out = np.full(n, fill, dtype=float)
    if len(series):
        out[series.index.to_numpy(dtype=int)] = series.to_numpy(dtype=float)
    return out


def _count(m: pd.DataFrame, n: int, mask: pd.Series | None = None) -> np.ndarray:
    sub = m if mask is None else m[mask]
    return _per_row(sub.groupby("row").size(), n, 0.0)


def _days(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86400.0


def build_features(records: Records, episodes: pd.DataFrame, cfg: dict[str, Any],
                   episode_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the feature matrix for `episodes` (rows of cohort/followup_episodes).

    `episode_history` holds the episode *definitions* used for the prior follow-up features
    (defaults to `episodes`). Only matured earlier episodes whose index event was recorded before
    t0 are used, and their completion is re-derived from records recorded before t0.
    """
    history = episodes if episode_history is None else episode_history
    ep = episodes[["episode_id", "patient_id", "programme", "prediction_at", "due_date"]].copy()
    ep = ep.reset_index(drop=True)
    ep["row"] = np.arange(len(ep))
    ep["t0"] = as_utc(ep["prediction_at"])
    ep["due"] = as_utc(ep["due_date"])
    n = len(ep)
    f: dict[str, np.ndarray] = {}
    enc_type = {name: p["encounter_type"] for name, p in cfg["programmes"].items()}
    ep["appointment_type"] = ep["programme"].map(enc_type)

    # ------------------------------------------------------------ attendance (appointments)
    ap = _utc(records.appointments, ["booked_at", "scheduled_for", "status_recorded_at"])
    m = _known(ep, ap.drop(columns=[]), "booked_at")
    m = m.rename(columns={"appointment_type": "appt_type"})
    m["status_t0"] = m["status"].where(m["status_recorded_at"] < m["t0"], "booked")
    win24 = m["scheduled_for"] >= m["t0"] - pd.Timedelta(days=S.LOOKBACK_DAYS)
    win12 = m["scheduled_for"] >= m["t0"] - pd.Timedelta(days=S.SHORT_WINDOW_DAYS)
    outcome = m["status_t0"].isin(OUTCOME_STATUSES) & win24
    attended = (m["status_t0"] == "attended") & win24
    noshow = (m["status_t0"] == "no_show") & win24
    f["prior_appointments_24m"] = _count(m, n, outcome)
    f["noshows_12m"] = _count(m, n, (m["status_t0"] == "no_show") & win12)
    f["noshows_24m"] = _count(m, n, noshow)
    n_att = _count(m, n, attended)
    f["attendance_rate_24m_smoothed"] = (
        (n_att + S.ATTENDANCE_PRIOR_STRENGTH * S.ATTENDANCE_PRIOR)
        / (n_att + f["noshows_24m"] + S.ATTENDANCE_PRIOR_STRENGTH))
    last_att = m[attended].groupby("row")["scheduled_for"].max()
    ns = m[noshow].copy()
    ns["last_att"] = ns["row"].map(last_att)
    after_last = ns["last_att"].isna() | (ns["scheduled_for"] > ns["last_att"])
    f["consecutive_noshows"] = _per_row(ns[after_last].groupby("row").size(), n, 0.0)
    f["patient_cancellations_12m"] = _count(m, n, (m["status_t0"] == "cancelled_patient") & win12)

    # booking for this follow-up: made before t0, slot at/after t0, still open at t0
    own = m[(m["appt_type"] == m["row"].map(ep["appointment_type"]))
            & (m["scheduled_for"] >= m["t0"]) & (m["status_t0"] == "booked")]
    own = own.sort_values(["row", "scheduled_for"]).groupby("row").first()
    f["has_open_booking"] = _per_row(pd.Series(1.0, index=own.index), n, 0.0)
    f["booking_lead_days"] = _per_row(_days(own["scheduled_for"] - own["booked_at"]), n)
    due_by_row = ep.set_index("row")["due"]
    f["booking_offset_from_due_days"] = _per_row(
        _days(own["scheduled_for"].dt.floor("D") - due_by_row.loc[own.index]), n)

    # ------------------------------------------------------------ encounters
    enc = _utc(records.encounters, ["start_at", "recorded_at"])
    me = _known(ep, enc, "recorded_at")
    last_visit = me.groupby("row")["start_at"].max()
    first_visit = me.groupby("row")["start_at"].min()
    t0_by_row = ep.set_index("row")["t0"]
    f["days_since_last_visit"] = _per_row(_days(t0_by_row.loc[last_visit.index] - last_visit), n)
    f["history_days"] = _per_row(_days(t0_by_row.loc[first_visit.index] - first_visit), n)
    acute12 = (me["encounter_type"] == "acute") & (
        me["start_at"] >= me["t0"] - pd.Timedelta(days=S.SHORT_WINDOW_DAYS))
    f["acute_visits_12m"] = _count(me, n, acute12)

    # ------------------------------------------------------------ prior follow-up outcomes
    f.update(_prior_followup(ep, records, cfg, history, n))

    # ------------------------------------------------------------ measurements
    obs = _utc(records.observations, ["effective_at", "recorded_at"])
    for prefix, key in (("hba1c", "hba1c"), ("sbp", "sbp")):
        code = cat.OBSERVATIONS[key][0]
        mo = _known(ep, obs[obs["code"] == code], "recorded_at", "effective_at", S.LOOKBACK_DAYS)
        mo = mo.sort_values(["row", "effective_at", "observation_id"])
        last = mo.groupby("row").last()
        f[f"{prefix}_last"] = _per_row(last["value_num"], n)
        if prefix == "hba1c":
            f["days_since_hba1c"] = _per_row(_days(last["t0"] - last["effective_at"]), n)
            f["hba1c_count_24m"] = _count(mo, n)
        else:
            f["days_since_sbp"] = _per_row(_days(last["t0"] - last["effective_at"]), n)
            f["sbp_count_24m"] = _count(mo, n)
        f[f"{prefix}_trend_per_year"] = _trend(mo, n)

    # ------------------------------------------------------------ medications
    ms = _utc(records.medication_statements, ["recorded_at", "end_recorded_at", "start_date", "end_date"])
    mm = _known(ep, ms, "recorded_at")
    stop_known = mm["end_recorded_at"].notna() & (mm["end_recorded_at"] < mm["t0"])
    active = ~stop_known
    ladder_codes = {name: {cat.MEDICATIONS[k][0] for k in cat.LADDERS[name]} for name in cat.LADDERS}
    prog_by_row = ep.set_index("row")["programme"]
    on_ladder = pd.Series([c in ladder_codes[p] for c, p in zip(mm["code"], mm["row"].map(prog_by_row),
                                                               strict=True)], index=mm.index, dtype=bool)
    f["active_medications"] = _count(mm, n, active)
    f["programme_medications"] = _count(mm, n, active & on_ladder)
    year_ago = mm["t0"] - pd.Timedelta(days=S.SHORT_WINDOW_DAYS)
    f["medication_starts_12m"] = _count(mm, n, mm["start_date"] >= year_ago)
    f["medication_stops_12m"] = _count(mm, n, stop_known & (mm["end_date"] >= year_ago))

    # ------------------------------------------------------------ outreach
    oa = _utc(records.outreach_attempts, ["attempted_at", "recorded_at"])
    mr = _known(ep, oa, "recorded_at", "attempted_at", S.LOOKBACK_DAYS)
    in12 = mr["attempted_at"] >= mr["t0"] - pd.Timedelta(days=S.SHORT_WINDOW_DAYS)
    phone = mr["channel"] == "phone"
    reached = mr["outcome"].isin(REACHED)
    f["outreach_calls_12m"] = _count(mr, n, phone & in12)
    f["outreach_reached_12m"] = _count(mr, n, phone & reached & in12)
    calls24, reached24 = _count(mr, n, phone), _count(mr, n, phone & reached)
    f["outreach_contact_rate_24m_smoothed"] = (
        (reached24 + S.CONTACT_PRIOR_STRENGTH * S.CONTACT_PRIOR) / (calls24 + S.CONTACT_PRIOR_STRENGTH))
    f["recall_letters_12m"] = _count(mr, n, (mr["channel"] == "letter") & in12)
    f["wrong_number_24m"] = (_count(mr, n, mr["outcome"] == "wrong_number") > 0).astype(float)
    last_or = mr.groupby("row")["attempted_at"].max()
    f["days_since_last_outreach"] = _per_row(_days(t0_by_row.loc[last_or.index] - last_or), n)

    # ------------------------------------------------------------ context
    pts = ep[["row", "patient_id", "t0"]].merge(
        records.patients[["patient_id", "birth_date", "deprivation_quintile"]], on="patient_id",
        how="left").set_index("row").sort_index()
    f["age_years"] = (_days(pts["t0"] - as_utc(pts["birth_date"])) / 365.25).to_numpy(dtype=float)
    f["deprivation_quintile"] = pts["deprivation_quintile"].to_numpy(dtype=float)
    cond = _utc(records.conditions, ["recorded_at"])
    programme_codes = {p["condition_code"] for p in cfg["programmes"].values()}
    mc = _known(ep, cond[cond["code"].isin(programme_codes)], "recorded_at")
    f["chronic_conditions"] = _per_row(mc.groupby("row")["code"].nunique(), n, 0.0)
    f["programme_is_htn"] = (ep["programme"] == "htn").to_numpy(dtype=float)

    out = ep[["episode_id", "patient_id", "programme", "prediction_at"]].copy()
    for feat in S.FEATURES:
        out[feat.name] = f[feat.name]
        if feat.indicator:
            out[S.indicator_name(feat.name)] = np.isnan(f[feat.name]).astype(float)
    return out[["episode_id", "patient_id", "programme", "prediction_at", *S.feature_columns()]]


def _trend(mo: pd.DataFrame, n: int) -> np.ndarray:
    """Least-squares slope per year; NaN unless >= 2 values spanning >= TREND_MIN_SPAN_DAYS."""
    if mo.empty:
        return np.full(n, np.nan)
    x = _days(mo["effective_at"] - mo["t0"]) / 365.25
    g = pd.DataFrame({"row": mo["row"], "x": x, "y": mo["value_num"].astype(float)})
    g["xx"], g["xy"] = g["x"] ** 2, g["x"] * g["y"]
    agg = g.groupby("row").agg(k=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxx=("xx", "sum"),
                               sxy=("xy", "sum"), xmin=("x", "min"), xmax=("x", "max"))
    denom = agg["k"] * agg["sxx"] - agg["sx"] ** 2
    slope = (agg["k"] * agg["sxy"] - agg["sx"] * agg["sy"]) / denom
    ok = (agg["k"] >= 2) & ((agg["xmax"] - agg["xmin"]) * 365.25 >= S.TREND_MIN_SPAN_DAYS) & (denom > 0)
    return _per_row(slope[ok], n)


def _prior_followup(ep: pd.DataFrame, records: Records, cfg: dict[str, Any], history: pd.DataFrame,
                    n: int) -> dict[str, np.ndarray]:
    ecfg = cfg["episodes"]
    q = qualifying_events(records, cfg, FAR_FUTURE)
    index_rec = q.drop_duplicates("event_ref").set_index("event_ref")["recorded_at"]
    h = history[["episode_id", "patient_id", "programme", "index_event_at", "index_event_ref",
                 "window_end_date"]].rename(columns={"episode_id": "prior_id"}).copy()
    h["index_event_at"] = as_utc(h["index_event_at"])
    h["window_end"] = as_utc(h["window_end_date"])
    h["index_rec"] = h["index_event_ref"].map(index_rec)
    h = h[h["index_rec"].notna()]
    matured_by = pd.Timedelta(days=ecfg["recording_lag_days"] + 1)
    m = ep[["row", "patient_id", "programme", "t0"]].merge(h, on=["patient_id", "programme"])
    m = m[(m["index_event_at"] < m["t0"]) & (m["index_rec"] < m["t0"])
          & (m["window_end"] + matured_by <= m["t0"])]
    # on time = a qualifying event recorded before t0, effective after the index review
    # (beyond the merge window) and no later than the end of the label window
    qq = q[["patient_id", "programme", "effective_at", "recorded_at"]]
    j = m[["row", "prior_id", "patient_id", "programme", "t0", "index_event_at", "window_end"]].merge(
        qq, on=["patient_id", "programme"])
    merge = pd.Timedelta(days=ecfg["merge_window_days"])
    hit = j[(j["recorded_at"] < j["t0"]) & (j["effective_at"] >= j["index_event_at"] + merge)
            & (j["effective_at"] < j["window_end"] + DAY)]
    ontime_ids = set(zip(hit["row"], hit["prior_id"], strict=True))
    m = m.assign(on_time=[(r, p) in ontime_ids for r, p in zip(m["row"], m["prior_id"], strict=True)])
    n_prior = _per_row(m.groupby("row").size(), n, 0.0)
    n_ontime = _per_row(m.groupby("row")["on_time"].sum(), n, 0.0)
    last = m.sort_values(["row", "index_event_at"]).groupby("row").last()
    return {
        "prior_episodes_matured": n_prior,
        "prior_ontime_rate_smoothed": (n_ontime + S.COMPLETION_PRIOR_STRENGTH * S.COMPLETION_PRIOR)
        / (n_prior + S.COMPLETION_PRIOR_STRENGTH),
        "last_prior_followup_late": _per_row((~last["on_time"]).astype(float), n),
    }
