"""Training, evaluation design, explanations and the saved inference bundle."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import read_table
from cura_ml.features import spec as S
from cura_ml.modeling.bundle import TIERS, load_bundle
from cura_ml.modeling.explain import contributions, native_tree_contributions
from cura_ml.modeling.models import raw_log_odds, raw_scores
from cura_ml.modeling.train import (
    build_feature_table,
    evaluate_bundle,
    predict_episodes,
    train_and_evaluate,
    training_table,
)

TRAIN_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_v1.yaml"


def _train_cfg(force: str) -> dict:
    cfg = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    cfg["evaluation"]["bootstrap_resamples"] = 40
    cfg["logistic_regression"]["C_grid"] = [0.01, 0.1]
    cfg["xgboost"]["max_depth_grid"] = [2, 3]
    cfg["xgboost"]["n_estimators_max"] = 300
    # force the selection branch so both bundle types are exercised
    cfg["selection"]["xgboost_pr_auc_ci_lower_bound_above"] = 10.0 if force == "lr" else -10.0
    cfg["selection"]["max_brier_increase_for_xgboost"] = 10.0
    cfg["model_version"] = f"test-{force}"
    return cfg


@pytest.fixture(scope="module")
def trained(small_dataset, tmp_path_factory):
    cohort_cfg, data_dir = small_dataset
    root = tmp_path_factory.mktemp("models")
    out = {}
    for force in ("lr", "xgb"):
        tcfg = _train_cfg(force)
        out[force] = (tcfg, train_and_evaluate(cohort_cfg, tcfg, data_dir, root, root / f"{force}.md"))
    return cohort_cfg, data_dir, out


@pytest.fixture(scope="module")
def table(trained):
    cohort_cfg, data_dir, out = trained
    feats = build_feature_table(cohort_cfg, data_dir, write=False)
    return training_table(cohort_cfg, out["lr"][0], data_dir, feats)


def test_selection_branches(trained):
    _, _, out = trained
    assert out["lr"][1]["metadata"]["model_type"] == "logistic_regression"
    assert out["xgb"][1]["metadata"]["model_type"] == "xgboost"


def test_split_design(trained, table):
    cohort_cfg, data_dir, out = trained
    sc = out["lr"][0]["splits"]
    by = {s: table[table["split"] == s] for s in ("train", "val_tune", "val_cal", "test")}
    assert all(len(v) for v in by.values())
    assert by["test"]["holdout_patient"].all()
    assert not table[table["split"].isin(["train", "val_tune", "val_cal"])]["holdout_patient"].any()
    assert (by["test"]["year"] == sc["test_year"]).all()
    assert set(by["train"]["year"]) <= set(sc["train_years"])
    assert (by["val_cal"]["year"] == sc["validation_year"]).all()
    assert (by["train"]["outcome_known_at"] < pd.Timestamp(f"{sc['validation_year']}-01-01", tz="UTC")).all()
    assert (by["val_tune"]["outcome_known_at"] < pd.Timestamp(f"{sc['test_year']}-01-01", tz="UTC")).all()
    pts = {s: set(v["patient_id"]) for s, v in by.items()}
    assert not pts["test"] & (pts["train"] | pts["val_tune"] | pts["val_cal"])
    assert not pts["val_tune"] & pts["val_cal"]


def test_features_and_labels_aligned(trained, table):
    _, data_dir, _ = trained
    labels = read_table(data_dir, "outcomes", "episode_labels").set_index("episode_id")
    assert table["episode_id"].is_unique
    ids = table["episode_id"]
    assert (table["label"].to_numpy() == labels.loc[ids, "label"].astype(int).to_numpy()).all()
    episodes = read_table(data_dir, "cohort", "followup_episodes").set_index("episode_id")
    assert (table["patient_id"].to_numpy() == episodes.loc[ids, "patient_id"].to_numpy()).all()
    assert (labels.loc[table["episode_id"], "label_status"] == "labelled").all()


def test_preprocessing_and_calibration_fitted_on_their_splits_only(trained, table):
    _, _, out = trained
    bundle = load_bundle(out["lr"][1]["bundle_dir"])
    assert isinstance(bundle.model, Pipeline)
    train = table[table["split"] == "train"][S.feature_columns()].to_numpy(float)
    imp, sc = bundle.model.named_steps["impute"], bundle.model.named_steps["scale"]
    np.testing.assert_allclose(imp.statistics_, np.nanmedian(train, axis=0))
    np.testing.assert_allclose(sc.mean_, imp.transform(train).mean(axis=0))
    everything = table[S.feature_columns()].to_numpy(float)
    assert not np.allclose(sc.mean_, imp.transform(everything).mean(axis=0))
    val_cal = table[table["split"] == "val_cal"]
    assert bundle.calibrator.n_fit_ == len(val_cal)
    probs = bundle.predict(val_cal)["probability"].to_numpy()
    tiers = out["lr"][0]["tiers"]
    assert bundle.metadata["tier_thresholds"]["high"] == pytest.approx(
        np.quantile(probs, 1 - tiers["high_fraction"]))


@pytest.mark.parametrize("kind", ["lr", "xgb"])
def test_explanations_add_up_and_name_real_features(trained, table, kind):
    _, _, out = trained
    bundle = load_bundle(out[kind][1]["bundle_dir"])
    test = table[table["split"] == "test"].head(25)
    X = bundle.matrix(test)
    contrib, base, method = contributions(bundle.model, X)
    np.testing.assert_allclose(base + contrib.sum(axis=1), raw_log_odds(bundle.model, X), atol=1e-4)
    if kind == "xgb":
        assert method == "tree_shap"
        native, _ = native_tree_contributions(bundle.model, X)
        np.testing.assert_allclose(contrib, native, atol=1e-5)
        assert isinstance(bundle.model, XGBClassifier)
    rows = bundle.explain(test, top_k=4)
    pred = bundle.predict(test)
    for row, (_, feat_row), p in zip(rows, test.iterrows(), pred.itertuples(), strict=True):
        assert row["episode_id"] == feat_row["episode_id"]
        assert row["model_log_odds"] == pytest.approx(p.model_log_odds, abs=1e-3)
        assert row["calibrated_probability"] == pytest.approx(p.probability, abs=1e-4)
        assert "not percentages" in row["units"]
        assert len(row["factors"]) == 4
        for f in row["factors"]:
            assert f["feature"] in S.FEATURE_BY_NAME
            v = feat_row[f["feature"]]
            assert f["recorded"] == (not np.isnan(v))
            if f["recorded"] and S.FEATURE_BY_NAME[f["feature"]].unit != "yes/no":
                assert f["value"] == pytest.approx(round(float(v), 2))
            assert (f["contribution_log_odds"] > 0) == f["direction"].startswith("raises")


def test_saved_bundle_contents_and_reload(trained):
    _, _, out = trained
    for kind in ("lr", "xgb"):
        d = Path(out[kind][1]["bundle_dir"])
        for name in ("model.joblib", "metadata.json", "feature_spec.json", "evaluation.json",
                     "model_card.md"):
            assert (d / name).exists(), name
        md = load_bundle(d).metadata
        for key in ("model_version", "feature_columns", "tier_thresholds", "training_config",
                    "data_fingerprint", "calibration_method", "library_versions"):
            assert key in md
        assert md["synthetic_data_only"] is True
    assert (Path(out["xgb"][1]["bundle_dir"]) / "xgboost_model.json").exists()
    card = (Path(out["lr"][1]["bundle_dir"]) / "model_card.md").read_text("utf-8")
    assert "not clinical validation" in card


def test_bundle_scores_open_eligible_episode(trained):
    _, data_dir, out = trained
    rows = predict_episodes(out["lr"][1]["bundle_dir"], data_dir, top_k=3)
    assert rows, "expected open eligible episodes at the cutoff"
    episodes = read_table(data_dir, "cohort", "followup_episodes").set_index("episode_id")
    labels = read_table(data_dir, "outcomes", "episode_labels").set_index("episode_id")
    for r in rows[:20]:
        assert 0.0 <= r["calibrated_probability"] <= 1.0
        assert r["risk_tier"] in TIERS
        assert episodes.loc[r["episode_id"], "eligible"]
        assert labels.loc[r["episode_id"], "label_status"] == "immature"
    one = predict_episodes(out["xgb"][1]["bundle_dir"], data_dir, [rows[0]["episode_id"]], top_k=3)
    assert one[0]["episode_id"] == rows[0]["episode_id"] and one[0]["method"] == "tree_shap"


def test_evaluate_reproduces_stored_metrics(trained):
    _, data_dir, out = trained
    for kind in ("lr", "xgb"):
        res = evaluate_bundle(out[kind][1]["bundle_dir"], data_dir)
        assert res["matches"], res
        assert res["data_fingerprint_matches"]


def test_training_is_reproducible(trained, tmp_path):
    cohort_cfg, data_dir, out = trained
    tcfg = copy.deepcopy(out["xgb"][0])
    again = train_and_evaluate(cohort_cfg, tcfg, data_dir, tmp_path, None)
    a, b = out["xgb"][1]["evaluation"]["test"]["point"], again["evaluation"]["test"]["point"]
    for model in a:
        for k in ("pr_auc", "roc_auc", "brier"):
            assert a[model][k] == b[model][k], (model, k)


def test_test_split_never_used_for_fitting(trained, table):
    """The test split is held-out patients in the test year only, and training used exactly the
    train split (the fitted-on-split checks for preprocessing/calibration are in a separate test)."""
    cohort_cfg, data_dir, out = trained
    ev = out["lr"][1]["evaluation"]
    assert ev["splits"]["test"]["episodes"] == int((table["split"] == "test").sum())
    assert ev["test_patients_overlapping_training_or_validation"] == 0
    bundle = load_bundle(out["lr"][1]["bundle_dir"])
    assert bundle.metadata["trained_on"]["episodes"] == int((table["split"] == "train").sum())
    test_years = as_utc(read_table(data_dir, "cohort", "followup_episodes").set_index("episode_id")
                        .loc[table.loc[table["split"] == "test", "episode_id"], "prediction_at"]).dt.year
    assert (test_years == out["lr"][0]["splits"]["test_year"]).all()
    assert raw_scores(bundle.model, bundle.matrix(table[table["split"] == "test"])).shape[0] == \
        ev["splits"]["test"]["episodes"]
