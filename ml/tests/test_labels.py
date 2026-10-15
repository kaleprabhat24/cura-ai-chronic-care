"""Hand-built fixtures for the episode, eligibility, censoring, maturity and label rules.

Reference T2D episode used by most cases (interval 182 days, lead 14, grace 30, lag 7):
    index review 2023-01-02 10:00 -> due 2023-07-03, t0 2023-06-19 00:00 UTC,
    window [2023-06-19, 2023-08-02], mature when cutoff >= 2023-08-09 (cutoff is 2024-12-31).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cura_ml.cohort.episodes import build_episodes
from cura_ml.dataset import Records
from cura_ml.synth import catalog as cat
from cura_ml.synth.generator import COLUMNS

INDEX = "2023-01-02 10:00"
CODES = {"t2d": cat.CONDITIONS["t2d"]["code"], "htn": cat.CONDITIONS["htn"]["code"]}


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


class Fixture:
    """Tiny DSL for building Records by hand."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {t: [] for t in COLUMNS}
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:05d}"

    def patient(self, pid: str, death: str | None = None, transfer: str | None = None,
                condition: str | None = "t2d", condition_recorded: str = "2022-06-01 09:00"):
        self.rows["patients"].append({
            "patient_id": pid, "synthetic_mrn": pid.upper(), "given_name": "Test",
            "family_name": "Fixture", "birth_date": date(1960, 1, 1), "sex": "female",
            "deprivation_quintile": 3, "preferred_language": "en", "registered_on": date(2010, 1, 1),
            "death_date": date.fromisoformat(death) if death else None,
            "death_recorded_at": ts(death) + pd.Timedelta(days=2) if death else pd.NaT,
            "transfer_out_date": date.fromisoformat(transfer) if transfer else None,
            "transfer_recorded_at": ts(transfer) + pd.Timedelta(hours=9) if transfer else pd.NaT,
            "is_synthetic": True,
        })
        if condition:
            self.condition(pid, condition, condition_recorded)
        return self

    def condition(self, pid: str, prog: str, recorded: str):
        self.rows["conditions"].append({
            "condition_id": self._id("cd"), "patient_id": pid, "code_system": cat.SNOMED,
            "code": CODES[prog], "display": prog, "onset_date": date(2015, 1, 1),
            "recorded_at": ts(recorded), "clinical_status": "active",
        })
        return self

    def encounter(self, pid: str, etype: str, start: str, appointment_id: str | None = None):
        start_ts = ts(start)
        eid = self._id("en")
        self.rows["encounters"].append({
            "encounter_id": eid, "patient_id": pid, "encounter_type": etype, "start_at": start_ts,
            "end_at": start_ts + pd.Timedelta(minutes=20), "status": "finished",
            "appointment_id": appointment_id, "recorded_at": start_ts + pd.Timedelta(minutes=30),
        })
        return eid

    def review(self, pid: str, start: str, prog: str = "t2d", with_appointment: bool = True):
        appt = None
        if with_appointment:
            appt = self.appointment(pid, start, "attended", prog)
        self.encounter(pid, f"review_{prog}", start, appt)
        return self

    def observation(self, pid: str, key: str, effective: str, recorded: str | None = None,
                    value: float = 7.0, encounter_id: str | None = None):
        code, display, unit, _ = cat.OBSERVATIONS[key]
        eff = ts(effective)
        self.rows["observations"].append({
            "observation_id": self._id("ob"), "patient_id": pid, "encounter_id": encounter_id,
            "code_system": cat.LOINC, "code": code, "display": display, "value_num": value,
            "unit": unit, "effective_at": eff,
            "recorded_at": ts(recorded) if recorded else eff + pd.Timedelta(days=2),
        })
        return self

    def appointment(self, pid: str, scheduled: str, status: str, prog: str = "t2d") -> str:
        sched = ts(scheduled)
        aid = self._id("ap")
        rec = sched + pd.Timedelta(minutes=20) if status in ("attended", "no_show") else \
            sched - pd.Timedelta(days=2)
        self.rows["appointments"].append({
            "appointment_id": aid, "patient_id": pid, "appointment_type": f"review_{prog}",
            "booked_at": sched - pd.Timedelta(days=20), "scheduled_for": sched, "status": status,
            "status_recorded_at": rec, "booking_source": "patient",
        })
        return aid

    def records(self) -> Records:
        frames = {}
        for t in Records.__dataclass_fields__:
            df = pd.DataFrame(self.rows[t], columns=COLUMNS[t])
            for col in df.columns:
                if col.endswith("_at") or col == "scheduled_for":
                    df[col] = pd.to_datetime(df[col], utc=True)
            frames[t] = df
        return Records(**frames)


def build(fx: Fixture, cfg: dict) -> pd.DataFrame:
    episodes, labels = build_episodes(fx.records(), cfg)
    return episodes.merge(labels, on="episode_id", validate="one_to_one")


def first_episode(df: pd.DataFrame, pid: str, prog: str = "t2d") -> pd.Series:
    rows = df[(df["patient_id"] == pid) & (df["programme"] == prog)].sort_values("index_event_at")
    assert len(rows), f"no episode for {pid}"
    return rows.iloc[0]


@pytest.fixture(scope="module")
def result(base_cfg) -> pd.DataFrame:
    fx = Fixture()
    # A: completion before t0 -> ineligible
    fx.patient("pA").review("pA", INDEX).review("pA", "2023-06-16 10:00")
    # B: completion on the t0 day -> eligible, label 0
    fx.patient("pB").review("pB", INDEX).review("pB", "2023-06-19 09:00")
    # C: completion on due + 30 days -> label 0
    fx.patient("pC").review("pC", INDEX).review("pC", "2023-08-02 16:00")
    # D: completion on due + 31 days -> label 1
    fx.patient("pD").review("pD", INDEX).review("pD", "2023-08-03 09:00")
    # E: never completed -> label 1
    fx.patient("pE").review("pE", INDEX)
    # F: lab-only HbA1c completes a T2D review (observation path)
    fx.patient("pF").review("pF", INDEX).observation("pF", "hba1c", "2023-07-10 08:00")
    # G: HTN — BP at an acute visit does not complete the review
    fx.patient("pG", condition="htn").review("pG", INDEX, prog="htn")
    eid = fx.encounter("pG", "acute", "2023-12-28 11:00")
    fx.observation("pG", "sbp", "2023-12-28 11:05", value=150, encounter_id=eid)
    # H: no-show and patient-cancelled review appointments do not complete
    fx.patient("pH").review("pH", INDEX)
    fx.appointment("pH", "2023-06-26 10:00", "no_show")
    fx.appointment("pH", "2023-07-20 10:00", "cancelled_patient")
    fx.appointment("pH", "2023-07-25 10:00", "cancelled_clinic")
    # I: attended review appointment without an encounter row still completes
    fx.patient("pI").review("pI", INDEX)
    fx.appointment("pI", "2023-07-05 10:00", "attended")
    # J: death mid-window without completion -> censored
    fx.patient("pJ", death="2023-07-15").review("pJ", INDEX)
    # K: completion in window, death afterwards -> labelled 0, not censored
    fx.patient("pK", death="2023-07-20").review("pK", INDEX).review("pK", "2023-07-01 10:00")
    # L: transfer out before t0 -> ineligible
    fx.patient("pL", transfer="2023-05-01").review("pL", INDEX)
    # M: window ends after cutoff - lag -> immature
    fx.patient("pM").review("pM", "2024-06-03 10:00")
    # N: t0 after the data cutoff -> not yet predictable
    fx.patient("pN").review("pN", "2024-08-01 10:00")
    # O: HbA1c effective in window but recorded after the cutoff -> not known, label 1
    fx.patient("pO").review("pO", INDEX)
    fx.observation("pO", "hba1c", "2023-07-10 08:00", recorded="2025-01-05 09:00")
    # P: condition first recorded after t0 -> ineligible
    fx.patient("pP", condition_recorded="2023-07-01 09:00").review("pP", INDEX)
    # Q: index event is a lab-only HbA1c and the patient has no encounters -> ineligible
    fx.patient("pQ").observation("pQ", "hba1c", INDEX)
    # R: events within the 14-day merge window belong to the same review
    fx.patient("pR").review("pR", INDEX).observation("pR", "hba1c", "2023-01-02 10:05")
    fx.observation("pR", "hba1c", "2023-01-10 09:00")
    # S: death on the t0 day -> ineligible
    fx.patient("pS", death="2023-06-19").review("pS", INDEX)
    # T: older encounter history -> not cold start (pE has only 167 full days of history -> cold start)
    fx.patient("pT").review("pT", INDEX)
    fx.encounter("pT", "acute", "2022-01-10 10:00")
    # U: maturity boundary — cutoff == window_end + lag (mature) vs one day later (immature)
    fx.patient("pU1").review("pU1", "2024-05-26 10:00")
    fx.patient("pU2").review("pU2", "2024-05-27 10:00")
    # V: transfer after completion in window -> labelled 0 (transfer does not censor)
    fx.patient("pV", transfer="2023-07-30").review("pV", INDEX).review("pV", "2023-07-10 10:00")
    # W: transfer mid-window without completion -> censored (transfer_out)
    fx.patient("pW", transfer="2023-07-10").review("pW", INDEX)
    return build(fx, base_cfg)


def test_reference_dates(result):
    e = first_episode(result, "pE")
    assert e["due_date"] == date(2023, 7, 3)
    assert e["prediction_at"] == ts("2023-06-19")
    assert e["window_end_date"] == date(2023, 8, 2)


@pytest.mark.parametrize("pid,reason", [
    ("pA", "completed_before_prediction"),
    ("pL", "transferred_before_prediction"),
    ("pN", "prediction_after_cutoff"),
    ("pP", "condition_not_recorded"),
    ("pQ", "no_prior_encounter"),
    ("pS", "deceased_before_prediction"),
])
def test_ineligible_reasons(result, pid, reason):
    e = first_episode(result, pid)
    assert not e["eligible"]
    assert e["ineligible_reason"] == reason
    assert e["label_status"] == "ineligible"
    assert pd.isna(e["label"])


@pytest.mark.parametrize("pid,label,completion_type", [
    ("pB", 0, "encounter"),
    ("pC", 0, "encounter"),
    ("pD", 1, "encounter"),
    ("pE", 1, None),
    ("pF", 0, "observation"),
    ("pH", 1, None),
    ("pI", 0, "appointment"),
    ("pK", 0, "encounter"),
    ("pO", 1, None),
    ("pV", 0, "encounter"),
])
def test_labels(result, pid, label, completion_type):
    e = first_episode(result, pid)
    assert e["eligible"] and e["label_status"] == "labelled"
    assert e["label"] == label
    assert e["completed_in_window"] == (label == 0)
    if completion_type is None:
        assert pd.isna(e["completed_at"]) or e["completed_at"] > ts("2023-08-03")
    else:
        assert e["completion_event_type"] == completion_type
    assert not e["censored"]


def test_htn_bp_observation_does_not_complete(result):
    e = first_episode(result, "pG", "htn")
    assert e["due_date"] == date(2024, 1, 2)
    assert e["label_status"] == "labelled" and e["label"] == 1
    assert pd.isna(e["completed_at"])


@pytest.mark.parametrize("pid,reason", [("pJ", "death"), ("pW", "transfer_out")])
def test_censoring(result, pid, reason):
    e = first_episode(result, pid)
    assert e["eligible"]
    assert e["censored"] and e["censor_reason"] == reason
    assert e["label_status"] == "censored" and pd.isna(e["label"])


def test_immature_episode_has_no_label(result):
    e = first_episode(result, "pM")
    assert e["eligible"] and not e["mature"]
    assert e["label_status"] == "immature" and pd.isna(e["label"])


def test_maturity_boundary(result):
    assert first_episode(result, "pU1")["mature"]
    assert first_episode(result, "pU1")["label_status"] == "labelled"
    assert not first_episode(result, "pU2")["mature"]


def test_merge_window_creates_single_review(result):
    rows = result[result["patient_id"] == "pR"]
    assert len(rows) == 1
    assert rows.iloc[0]["index_event_type"] == "encounter"


def test_completion_starts_next_episode(result):
    rows = result[result["patient_id"] == "pC"].sort_values("index_event_at")
    assert len(rows) == 2
    assert rows.iloc[1]["index_event_at"] == rows.iloc[0]["completed_at"]


def test_cold_start_flag(result):
    assert first_episode(result, "pE")["history_days"] == 167
    assert first_episode(result, "pE")["is_cold_start"]
    assert not first_episode(result, "pT")["is_cold_start"]
