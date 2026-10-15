"""Event-time consistency, units and referential integrity of the generated records."""

from __future__ import annotations

import pandas as pd
import pytest

from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import load_records
from cura_ml.synth import catalog as cat


@pytest.fixture(scope="module")
def rec(small_dataset):
    return load_records(small_dataset[1])


@pytest.fixture(scope="module")
def start(small_dataset):
    return pd.Timestamp(small_dataset[0]["simulation"]["start_date"], tz="UTC")


def test_appointment_times(rec, cutoff_end):
    ap = rec.appointments
    booked, sched, rec_at = (as_utc(ap[c]) for c in ("booked_at", "scheduled_for",
                                                    "status_recorded_at"))
    assert (booked <= sched).all()
    assert (booked < cutoff_end).all()
    assert (rec_at < cutoff_end).all()
    done = ap["status"].isin(["attended", "no_show"])
    assert (rec_at[done] >= sched[done]).all()
    cancelled = ap["status"].str.startswith("cancelled")
    assert ((booked[cancelled] <= rec_at[cancelled]) & (rec_at[cancelled] <= sched[cancelled])).all()
    assert set(ap["status"]) <= set(cat.APPOINTMENT_STATUSES)


def test_future_appointments_are_only_scheduling_information(rec, cutoff_end):
    """Appointments after the cutoff appear only as bookings (or cancellations) recorded before the
    cutoff — never as attended / no-show outcomes."""
    ap = rec.appointments
    future = as_utc(ap["scheduled_for"]) >= cutoff_end
    assert (ap.loc[future, "status"] == "booked").any(), "expected some booked future appointments"
    assert ap.loc[future, "status"].isin(["booked", "cancelled_patient", "cancelled_clinic"]).all()
    booked = ap["status"] == "booked"
    assert (as_utc(ap.loc[booked, "scheduled_for"]) >= cutoff_end).all()
    assert (as_utc(ap.loc[booked, "status_recorded_at"]) == as_utc(ap.loc[booked, "booked_at"])).all()


def test_no_clinical_events_in_the_future(rec, cutoff_end):
    for table, cols in (
        ("encounters", ["start_at", "end_at", "recorded_at"]),
        ("observations", ["effective_at", "recorded_at"]),
        ("conditions", ["recorded_at"]),
        ("medication_statements", ["recorded_at"]),
        ("outreach_attempts", ["attempted_at", "recorded_at"]),
    ):
        df = getattr(rec, table)
        for c in cols:
            assert (as_utc(df[c]) < cutoff_end).all(), f"{table}.{c} after cutoff"


def test_effective_before_recorded(rec, start):
    enc = rec.encounters
    s, e, r = as_utc(enc["start_at"]), as_utc(enc["end_at"]), as_utc(enc["recorded_at"])
    assert ((s >= start) & (s < e) & (e <= r)).all()
    obs = rec.observations
    assert (as_utc(obs["effective_at"]) <= as_utc(obs["recorded_at"])).all()
    joined = obs.merge(enc[["encounter_id", "start_at"]], on="encounter_id", how="left")
    assert joined["start_at"].notna().all()
    assert (as_utc(joined["effective_at"]) >= as_utc(joined["start_at"])).all()
    oa = rec.outreach_attempts
    assert (as_utc(oa["attempted_at"]) <= as_utc(oa["recorded_at"])).all()
    cond = rec.conditions
    assert (as_utc(cond["onset_date"]) <= as_utc(cond["recorded_at"])).all()
    ms = rec.medication_statements
    assert (as_utc(ms["start_date"]) <= as_utc(ms["recorded_at"])).all()
    ended = ms["end_date"].notna()
    assert (as_utc(ms.loc[ended, "end_date"]) >= as_utc(ms.loc[ended, "start_date"])).all()
    assert (as_utc(ms.loc[ended, "end_date"]) <= as_utc(ms.loc[ended, "end_recorded_at"])).all()
    assert ms.loc[~ended, "end_recorded_at"].isna().all()


def test_no_events_after_death_or_transfer(rec):
    pts = rec.patients
    exits = pd.DataFrame({
        "patient_id": pts["patient_id"],
        "exit": as_utc(pts["death_date"]).fillna(as_utc(pts["transfer_out_date"])),
    }).dropna()
    assert len(exits), "expected some deaths/transfers in the test cohort"
    assert (as_utc(pts["death_date"]).dropna() <= as_utc(pts["death_recorded_at"]).dropna()).all()
    checks = (
        ("encounters", rec.encounters, "start_at"),
        ("observations", rec.observations, "effective_at"),
        ("outreach_attempts", rec.outreach_attempts, "attempted_at"),
        ("medication_statements", rec.medication_statements, "recorded_at"),
        ("attended appointments", rec.appointments[rec.appointments["status"] == "attended"],
         "scheduled_for"),
    )
    for name, df, col in checks:
        m = df.merge(exits, on="patient_id")
        assert (as_utc(m[col]) < m["exit"]).all(), f"{name} after exit"


def test_attended_reviews_have_matching_encounters(rec):
    att = rec.appointments[rec.appointments["status"] == "attended"]
    enc = rec.encounters[rec.encounters["appointment_id"].notna()]
    m = att.merge(enc, on="appointment_id", how="left", suffixes=("", "_enc"))
    assert m["encounter_id"].notna().all()
    assert (as_utc(m["scheduled_for"]) == as_utc(m["start_at"])).all()
    assert (m["appointment_type"] == m["encounter_type"]).all()


def test_units_codes_and_values(rec):
    obs = rec.observations
    assert (obs["code_system"] == cat.LOINC).all()
    assert set(obs["code"]) <= set(cat.UNIT_BY_LOINC)
    assert (obs["unit"] == obs["code"].map(cat.UNIT_BY_LOINC)).all()
    assert obs["value_num"].notna().all()
    for code, (lo, hi) in cat.RANGE_BY_LOINC.items():
        v = obs.loc[obs["code"] == code, "value_num"]
        assert len(v) and v.between(lo, hi).all(), code
    assert set(rec.medication_statements["code"]) <= {c for c, _, _ in cat.MEDICATIONS.values()}


def test_ids_unique_and_references_valid(rec):
    pids = set(rec.patients["patient_id"])
    assert rec.patients["patient_id"].is_unique
    assert rec.patients["is_synthetic"].all()
    for table, id_col in (
        ("conditions", "condition_id"), ("appointments", "appointment_id"),
        ("encounters", "encounter_id"), ("observations", "observation_id"),
        ("medication_statements", "medication_statement_id"), ("outreach_attempts", "outreach_id"),
    ):
        df = getattr(rec, table)
        assert df[id_col].is_unique, table
        assert set(df["patient_id"]) <= pids, table
