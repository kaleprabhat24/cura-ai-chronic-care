"""Leakage prevention and alignment tests for the feature builder."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import Records, load_records, read_table
from cura_ml.features import build as fb
from cura_ml.features import spec as S
from cura_ml.features.build import build_features, records_as_of
from cura_ml.synth import catalog as cat

from .test_labels import INDEX, Fixture

N_TRUNCATION_SAMPLES = 80


@pytest.fixture(scope="module")
def data(small_dataset):
    cfg, d = small_dataset
    rec = load_records(d)
    eps = read_table(d, "cohort", "followup_episodes")
    eligible = eps[eps["eligible"]].reset_index(drop=True)
    feats = build_features(rec, eligible, cfg, eps)
    return cfg, rec, eps, eligible, feats


def _patient_records(rec: Records, pid: str) -> Records:
    return Records(**{t: getattr(rec, t)[getattr(rec, t)["patient_id"] == pid].copy()
                      for t in Records.__dataclass_fields__})


def _sample(eligible: pd.DataFrame, k: int) -> pd.DataFrame:
    return eligible.sample(n=min(k, len(eligible)), random_state=7)


def _assert_rows_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    cols = S.feature_columns()
    pd.testing.assert_frame_equal(a[["episode_id", *cols]].reset_index(drop=True),
                                  b[["episode_id", *cols]].reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-9, check_dtype=False)


def test_features_identical_on_data_truncated_at_t0(data):
    """Recomputing each episode's features from the records as known at its t0 gives the same values
    as computing from the complete dataset."""
    cfg, rec, eps, eligible, feats = data
    sample = _sample(eligible, N_TRUNCATION_SAMPLES)
    assert sample["programme"].nunique() == 2
    by_id = feats.set_index("episode_id")
    for _, row in sample.iterrows():
        t0 = as_utc(pd.Series([row["prediction_at"]])).iloc[0]
        truncated = records_as_of(_patient_records(rec, row["patient_id"]), t0)
        got = build_features(truncated, row.to_frame().T, cfg, eps)
        _assert_rows_equal(got, by_id.loc[[row["episode_id"]]].reset_index())


def test_late_recorded_and_future_events_do_not_change_features(data):
    """Inject events that happened before t0 but were recorded after t0 (and events after t0):
    features must not move."""
    cfg, rec, eps, eligible, feats = data
    sample = _sample(eligible, 40)
    obs, ap, oa = [], [], []
    hba1c, _, unit, _ = cat.OBSERVATIONS["hba1c"]
    for i, row in enumerate(sample.itertuples()):
        t0 = as_utc(pd.Series([row.prediction_at])).iloc[0]
        late = t0 + pd.Timedelta(hours=1)
        obs.append({"observation_id": f"ob_poison_{i}", "patient_id": row.patient_id,
                    "encounter_id": None, "code_system": cat.LOINC, "code": hba1c, "display": "x",
                    "value_num": 14.9, "unit": unit, "effective_at": t0 - pd.Timedelta(days=10),
                    "recorded_at": late})
        for j, (sched, status) in enumerate(((t0 - pd.Timedelta(days=20), "no_show"),
                                             (t0 + pd.Timedelta(days=3), "attended"))):
            ap.append({"appointment_id": f"ap_poison_{i}_{j}", "patient_id": row.patient_id,
                       "appointment_type": f"review_{row.programme}",
                       "booked_at": sched - pd.Timedelta(days=30) if j == 0 else late,
                       "scheduled_for": sched, "status": status,
                       "status_recorded_at": max(sched + pd.Timedelta(minutes=20), late),
                       "booking_source": "patient"})
        oa.append({"outreach_id": f"or_poison_{i}", "patient_id": row.patient_id,
                   "programme": row.programme, "purpose": "post_due_chase", "channel": "phone",
                   "attempted_at": t0 - pd.Timedelta(days=2), "outcome": "wrong_number",
                   "next_action": "retry", "next_action_due": None, "performed_by": "x",
                   "recorded_at": late})

    def add(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
        extra = pd.DataFrame(rows)
        for c in extra.columns:
            if c.endswith("_at") or c == "scheduled_for":
                extra[c] = as_utc(extra[c])
        return pd.concat([df, extra[df.columns]], ignore_index=True)

    poisoned = Records(**{**{t: getattr(rec, t) for t in Records.__dataclass_fields__},
                          "observations": add(rec.observations, obs),
                          "appointments": add(rec.appointments, ap),
                          "outreach_attempts": add(rec.outreach_attempts, oa)})
    _assert_rows_equal(build_features(poisoned, sample, cfg, eps),
                       feats.set_index("episode_id").loc[sample["episode_id"]].reset_index())


def test_prior_followup_uses_only_matured_outcomes(base_cfg):
    fx = Fixture()
    # pX: early second review -> at the 2nd episode's t0 the 1st episode has not matured yet
    fx.patient("pX").review("pX", INDEX).review("pX", "2023-02-20 10:00")
    # pY: 1st episode completed late (after due + 30) -> matured, late
    fx.patient("pY").review("pY", INDEX).review("pY", "2023-08-20 10:00")
    # pZ: 1st episode completed on time -> matured, on time
    fx.patient("pZ").review("pZ", INDEX).review("pZ", "2023-07-01 10:00")
    from cura_ml.cohort.episodes import build_episodes

    rec = fx.records()
    eps, _ = build_episodes(rec, base_cfg)
    second = eps.sort_values("index_event_at").groupby("patient_id").nth(1).set_index("patient_id")
    f = build_features(rec, second.reset_index(), base_cfg, eps).set_index("patient_id")
    assert f.loc["pX", "prior_episodes_matured"] == 0
    assert np.isnan(f.loc["pX", "last_prior_followup_late"])
    assert f.loc["pX", "last_prior_followup_late__missing"] == 1
    assert f.loc["pY", "prior_episodes_matured"] == 1
    assert f.loc["pY", "last_prior_followup_late"] == 1
    assert f.loc["pY", "prior_ontime_rate_smoothed"] == pytest.approx((0 + 2 * 0.7) / 3)
    assert f.loc["pZ", "last_prior_followup_late"] == 0
    assert f.loc["pZ", "prior_ontime_rate_smoothed"] == pytest.approx((1 + 2 * 0.7) / 3)


def test_prior_episode_counts_match_independent_maturity_rule(data):
    cfg, _, eps, eligible, feats = data
    lag = cfg["episodes"]["recording_lag_days"]
    h = eps.assign(t_end=as_utc(eps["window_end_date"]) + pd.Timedelta(days=lag + 1))
    sample = _sample(eligible, 200)
    for row in sample.itertuples():
        t0 = as_utc(pd.Series([row.prediction_at])).iloc[0]
        same = h[(h["patient_id"] == row.patient_id) & (h["programme"] == row.programme)]
        expected = int((same["t_end"] <= t0).sum())
        got = feats.loc[feats["episode_id"] == row.episode_id, "prior_episodes_matured"].iloc[0]
        assert got == expected


def test_hidden_traits_never_enter_feature_matrix(small_dataset, data):
    _, d = small_dataset
    _, _, _, _, feats = data
    traits = read_table(d, "hidden", "simulation_traits")
    assert not set(traits.columns) - {"patient_id"} & set(feats.columns)
    source = "".join(inspect.getsource(m) for m in (fb, S))
    for token in ('"hidden"', "simulation_traits", "simulator_episode_log", "episode_labels",
                  "read_table", "load_records"):
        assert token not in source, token
    # sanity: no feature is a near-copy of a latent trait
    joined = feats.merge(traits, on="patient_id")
    for col in S.feature_columns():
        for trait in ("engagement", "access_barrier", "trust", "hba1c_baseline", "sbp_baseline"):
            r = joined[[col, trait]].dropna().corr(method="spearman").iloc[0, 1]
            assert np.isnan(r) or abs(r) < 0.9, (col, trait, r)


def test_features_aligned_to_episode_and_patient(data):
    cfg, rec, eps, eligible, feats = data
    assert feats["episode_id"].is_unique
    assert list(feats["episode_id"]) == list(eligible["episode_id"])
    assert (feats["patient_id"].to_numpy() == eligible["patient_id"].to_numpy()).all()
    assert (feats["programme"].to_numpy() == eligible["programme"].to_numpy()).all()
    assert list(feats.columns[4:]) == S.feature_columns()
    shuffled = eligible.sample(frac=1.0, random_state=3)
    again = build_features(rec, shuffled, cfg, eps).set_index("episode_id").loc[feats["episode_id"]]
    _assert_rows_equal(again.reset_index(), feats)


def test_missing_indicators_match_values(data):
    feats = data[4]
    for f in S.FEATURES:
        if f.indicator:
            assert (feats[S.indicator_name(f.name)] == feats[f.name].isna().astype(float)).all()
        elif f.name not in ("booking_lead_days", "booking_offset_from_due_days", "days_since_hba1c",
                            "days_since_sbp", "history_days"):
            assert feats[f.name].notna().all(), f.name
    no_booking = feats["has_open_booking"] == 0
    assert feats.loc[no_booking, "booking_lead_days"].isna().all()
    assert feats.loc[~no_booking, "booking_lead_days"].notna().all()


def test_feature_spec_documents_every_column():
    for f in S.FEATURES:
        for attr in ("label", "unit", "source", "window", "aggregation", "missing_policy"):
            assert getattr(f, attr), (f.name, attr)
    assert len(set(S.feature_columns())) == len(S.feature_columns())
    assert "| `hba1c_last` |" in S.spec_markdown()
