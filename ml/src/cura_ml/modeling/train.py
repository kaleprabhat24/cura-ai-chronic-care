"""Train, calibrate, evaluate, select and save the follow-up risk model.

Data flow (no step reads the test split before the final evaluation):
    train     -> fit models (preprocessing is part of each model and fitted here only)
    val_tune  -> choose LR regularisation and XGBoost depth / number of trees
    val_cal   -> fit calibrators, compare models, choose final model, fix risk-tier thresholds
    test      -> final evaluation on held-out patients in the latest year (reported, never tuned on)
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score

from cura_ml import __version__
from cura_ml.cohort.episodes import as_utc
from cura_ml.config import config_hash
from cura_ml.dataset import content_hash, load_records, read_table
from cura_ml.features import spec as S
from cura_ml.features.build import build_features
from cura_ml.manifest import read_manifest
from cura_ml.modeling import metrics as M
from cura_ml.modeling.bundle import assign_tiers, load_bundle, save_bundle
from cura_ml.modeling.explain import contributions, global_importance, native_tree_contributions
from cura_ml.modeling.models import (
    AttendanceHeuristic,
    Calibrator,
    PrevalenceBaseline,
    choose_calibrator,
    make_logistic_regression,
    make_xgboost,
    raw_scores,
)
from cura_ml.modeling.report import model_card, write_model_report
from cura_ml.modeling.splits import assign_splits

MODEL_NAMES = ("prevalence_baseline", "attendance_heuristic", "logistic_regression", "xgboost")
FEATURES_FILE = Path("features") / "episode_features.parquet"


def data_fingerprint(manifest: dict[str, Any]) -> str:
    parts = {"config_sha256": manifest["config_sha256"],
             **{k: v["content_sha256"] for k, v in sorted(manifest["files"].items())
                if not k.startswith("hidden/")}}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def build_feature_table(cohort_cfg: dict[str, Any], data_dir: str | Path,
                        write: bool = True) -> pd.DataFrame:
    """Features for every eligible episode (labelled or not), from ordinary records only."""
    records = load_records(data_dir)
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    feats = build_features(records, episodes[episodes["eligible"]], cohort_cfg, episodes)
    if write:
        path = Path(data_dir) / FEATURES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)
    return feats


def training_table(cohort_cfg: dict[str, Any], train_cfg: dict[str, Any], data_dir: str | Path,
                   feats: pd.DataFrame) -> pd.DataFrame:
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    labels = read_table(data_dir, "outcomes", "episode_labels")
    splits = assign_splits(episodes, labels, cohort_cfg, train_cfg)
    return splits.merge(feats, on=["episode_id", "patient_id", "programme"], how="inner",
                        validate="one_to_one")


def _xy(table: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    sub = table[table["split"] == split].reset_index(drop=True)
    return (sub, sub[S.feature_columns()].to_numpy(dtype=float), sub["label"].to_numpy(dtype=int),
            sub["patient_id"].to_numpy())


def train_and_evaluate(cohort_cfg: dict[str, Any], train_cfg: dict[str, Any], data_dir: str | Path,
                       bundle_root: str | Path, report_path: str | Path | None) -> dict[str, Any]:
    data_dir = Path(data_dir)
    seed = train_cfg["seed"]
    ev = train_cfg["evaluation"]
    cap = train_cfg["capacity"]
    cols = S.feature_columns()
    manifest = read_manifest(data_dir)
    feats = build_feature_table(cohort_cfg, data_dir)
    table = training_table(cohort_cfg, train_cfg, data_dir, feats)
    tr, Xtr, ytr, _ = _xy(table, "train")
    vt, Xvt, yvt, _ = _xy(table, "val_tune")
    vc, Xvc, yvc, gvc = _xy(table, "val_cal")
    te, Xte, yte, gte = _xy(table, "test")

    # ---------------------------------------------------------------- fit + tune (train / val_tune)
    models: dict[str, Any] = {
        "prevalence_baseline": PrevalenceBaseline().fit(Xtr, ytr),
        "attendance_heuristic": AttendanceHeuristic(cols.index("attendance_rate_24m_smoothed")),
    }
    lr_grid = []
    best_lr, best_lr_score = None, -1.0
    for C in train_cfg["logistic_regression"]["C_grid"]:
        m = make_logistic_regression(C).fit(Xtr, ytr)
        s = float(average_precision_score(yvt, raw_scores(m, Xvt)))
        lr_grid.append({"C": C, "val_tune_pr_auc": s})
        if s > best_lr_score:
            best_lr, best_lr_score = m, s
    models["logistic_regression"] = best_lr

    xcfg = train_cfg["xgboost"]
    xgb_grid = []
    for depth in xcfg["max_depth_grid"]:
        m = make_xgboost(xcfg, depth, xcfg["n_estimators_max"], seed, xcfg["early_stopping_rounds"])
        m.fit(Xtr, ytr, eval_set=[(Xvt, yvt)], verbose=False)
        xgb_grid.append({"max_depth": depth, "n_estimators": int(m.best_iteration) + 1,
                         "val_tune_pr_auc": float(m.best_score)})
    best_x = max(xgb_grid, key=lambda r: (r["val_tune_pr_auc"], -r["max_depth"]))
    # refit without early stopping so the saved model has a fixed, explicit number of trees
    models["xgboost"] = make_xgboost(xcfg, best_x["max_depth"], best_x["n_estimators"], seed).fit(Xtr, ytr)

    # ---------------------------------------------------------------- calibrate (val_cal)
    calibrators: dict[str, Calibrator] = {"prevalence_baseline": Calibrator("identity").fit(
        raw_scores(models["prevalence_baseline"], Xvc), yvc)}
    calibration_cv: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES[1:]:
        calibrators[name], calibration_cv[name] = choose_calibrator(
            raw_scores(models[name], Xvc), yvc, gvc, train_cfg["calibration"]["methods"],
            train_cfg["calibration"]["cv_folds"])

    def outputs(X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for name in MODEL_NAMES:
            s = raw_scores(models[name], X)
            out[name] = (s, calibrators[name].transform(s))
        return out

    val_out, test_out = outputs(Xvc), outputs(Xte)
    val_metrics = {n: M.point_metrics(yvc, *val_out[n], cap["report_fractions"], ev["calibration_bins"])
                   for n in MODEL_NAMES}

    # ---------------------------------------------------------------- select (val_cal only)
    sel = train_cfg["selection"]
    gain = val_metrics["xgboost"]["pr_auc"] - val_metrics["logistic_regression"]["pr_auc"]
    brier_diff = val_metrics["xgboost"]["brier"] - val_metrics["logistic_regression"]["brier"]
    val_boot = M.bootstrap(yvc, gvc, {k: val_out[k] for k in ("xgboost", "logistic_regression")},
                           cap["primary_fraction"], ev["bootstrap_resamples"], seed,
                           [("xgboost", "logistic_regression")])
    gain_ci = val_boot["differences"]["xgboost - logistic_regression"]["pr_auc"]["interval"]
    selected = ("xgboost" if gain_ci[0] > sel["xgboost_pr_auc_ci_lower_bound_above"]
                and brier_diff <= sel["max_brier_increase_for_xgboost"] else "logistic_regression")
    selection = {"rule": ("choose XGBoost only if the paired patient-level bootstrap 95% CI of its "
                          "validation PR-AUC advantage over logistic regression lies entirely above "
                          f"{sel['xgboost_pr_auc_ci_lower_bound_above']} and its validation Brier score is "
                          f"not higher by more than {sel['max_brier_increase_for_xgboost']}; otherwise "
                          "keep the simpler logistic regression"),
                 "xgboost_minus_lr_val_pr_auc": gain, "xgboost_minus_lr_val_pr_auc_ci": gain_ci,
                 "xgboost_minus_lr_val_brier": brier_diff, "selected": selected}
    val_prob = val_out[selected][1]
    tiers_cfg = train_cfg["tiers"]
    thresholds = {
        "high": float(np.quantile(val_prob, 1 - tiers_cfg["high_fraction"])),
        "medium": float(np.quantile(val_prob, 1 - tiers_cfg["high_fraction"] - tiers_cfg["medium_fraction"])),
    }

    # ---------------------------------------------------------------- final evaluation (test)
    test_metrics = {n: M.point_metrics(yte, *test_out[n], cap["report_fractions"], ev["calibration_bins"])
                    for n in MODEL_NAMES}
    pairs = [("xgboost", "logistic_regression"), ("xgboost", "attendance_heuristic"),
             ("logistic_regression", "attendance_heuristic"), ("logistic_regression", "prevalence_baseline")]
    boot = M.bootstrap(yte, gte, test_out, cap["primary_fraction"], ev["bootstrap_resamples"], seed, pairs)
    test_tiers = assign_tiers(test_out[selected][1], thresholds)
    tier_table = [{"tier": t, "n": int((test_tiers == t).sum()),
                   "share": float((test_tiers == t).mean()),
                   "observed_incomplete_rate": float(yte[test_tiers == t].mean()) if (test_tiers == t).any()
                   else float("nan")} for t in ("high", "medium", "low")]

    # ---------------------------------------------------------------- explanations + diagnostics
    importance, shap_check = {}, {}
    for name in ("logistic_regression", "xgboost"):
        contrib, base, method = contributions(models[name], Xte)
        importance[name] = {"method": method, "features": global_importance(contrib, cols)}
        if name == "xgboost":
            native, native_base = native_tree_contributions(models[name], Xte)
            margin = models[name].predict(Xte, output_margin=True)
            shap_check = {"max_abs_diff_shap_vs_xgboost_native": float(np.max(np.abs(native - contrib))),
                          "max_abs_additivity_error": float(np.max(np.abs(base + contrib.sum(1) - margin)))}
    leak_scan = M.single_feature_auc(Xtr, ytr, cols)
    patients = load_records(data_dir).patients[["patient_id", "sex"]]
    subgroups = _subgroups(te.merge(patients, on="patient_id", how="left"), *test_out[selected],
                           ev["calibration_bins"])
    missingness = {split: {c: float(np.isnan(table.loc[table["split"] == split, c].to_numpy(float)).mean())
                           for c in cols if not c.endswith("__missing")}
                   for split in ("train", "test")}
    prevalence = _prevalence_over_time(data_dir)

    split_summary = {s: {"episodes": int((table["split"] == s).sum()),
                         "patients": int(table.loc[table["split"] == s, "patient_id"].nunique()),
                         "prevalence": float(table.loc[table["split"] == s, "label"].mean())}
                     for s in ("train", "val_tune", "val_cal", "test")}
    overlap = set(table.loc[table["split"] == "test", "patient_id"]) & set(
        table.loc[table["split"].isin(["train", "val_tune", "val_cal"]), "patient_id"])

    evaluation = {
        "synthetic_data_only": True,
        "selected_model": selected, "selection": selection,
        "splits": split_summary, "test_patients_overlapping_training_or_validation": len(overlap),
        "tuning": {"logistic_regression": lr_grid, "xgboost": xgb_grid,
                   "logistic_regression_best_C": models["logistic_regression"].named_steps["clf"].C,
                   "xgboost_best": best_x},
        "calibration": {"cv_brier": calibration_cv,
                        "chosen": {n: calibrators[n].method for n in MODEL_NAMES}},
        "validation": val_metrics,
        "test": {"point": test_metrics, "bootstrap": boot,
                 "calibration_table": M.calibration_table(yte, test_out[selected][1], ev["calibration_bins"]),
                 "tiers": tier_table},
        "tier_thresholds": thresholds,
        "feature_importance": importance, "shap_consistency": shap_check,
        "single_feature_auc_train": leak_scan[:10],
        "suspicious_features": [f for f, a in leak_scan if a >= ev["suspicious_single_feature_auc"]],
        "subgroups_test": subgroups, "feature_missingness": missingness,
        "label_prevalence_over_time": prevalence,
    }

    version = train_cfg["model_version"]
    metadata = {
        "model_version": version, "model_type": selected, "cura_ml_version": __version__,
        "feature_spec_version": S.FEATURE_SPEC_VERSION, "feature_columns": cols,
        "calibration_method": calibrators[selected].method, "tier_thresholds": thresholds,
        "risk_tier_fractions": tiers_cfg,
        "hyperparameters": ({"C": models[selected].named_steps["clf"].C}
                            if selected == "logistic_regression" else best_x),
        "training_config": train_cfg, "training_config_sha256": config_hash(train_cfg),
        "cohort_config": cohort_cfg, "cohort_config_sha256": config_hash(cohort_cfg),
        "data_fingerprint": data_fingerprint(manifest), "data_dir": str(data_dir).replace("\\", "/"),
        "features_content_sha256": content_hash(feats),
        "trained_on": {"split": "train", "episodes": split_summary["train"]["episodes"],
                       "years": train_cfg["splits"]["train_years"]},
        "calibrated_on": {"split": "val_cal", "episodes": split_summary["val_cal"]["episodes"]},
        "explanation_method": ("logistic_regression_contributions" if selected == "logistic_regression"
                               else "tree_shap"),
        "prediction_target": ("follow-up still incomplete 30 days after the due date, predicted 14 days "
                              "before the due date"),
        "synthetic_data_only": True,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                             "xgboost": xgb.__version__, "shap": shap.__version__,
                             "joblib": joblib.__version__},
    }
    bundle_dir = Path(bundle_root) / version
    save_bundle(bundle_dir, models[selected], calibrators[selected], metadata, evaluation,
                model_card(metadata, evaluation))

    # example explanation from the saved bundle (proves the on-disk model works end to end)
    bundle = load_bundle(bundle_dir)
    top = te.iloc[[int(np.argmax(test_out[selected][0]))]]
    evaluation["example_explanation"] = bundle.explain(top, top_k=6)[0]
    (bundle_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if report_path is not None:
        write_model_report(report_path, metadata, evaluation)
    return {"bundle_dir": bundle_dir, "metadata": metadata, "evaluation": evaluation}


def _subgroups(df: pd.DataFrame, score: np.ndarray, prob: np.ndarray, bins: int) -> list[dict[str, Any]]:
    df = df.assign(score=score, prob=prob,
                   deprivation_group=pd.cut(df["deprivation_quintile"], [0, 2, 3, 5],
                                            labels=["1-2", "3", "4-5"]).astype(str))
    rows = []
    for col in ("programme", "sex", "deprivation_group"):
        for value, g in df.groupby(col, observed=True):
            y = g["label"].to_numpy()
            rows.append({"group": col, "value": str(value), "n": int(len(g)), "prevalence": float(y.mean()),
                         "pr_auc": float(average_precision_score(y, g["score"])) if 0 < y.sum() < len(y)
                         else float("nan"),
                         "mean_predicted": float(g["prob"].mean())})
    return rows


def _prevalence_over_time(data_dir: Path) -> list[dict[str, Any]]:
    ep = read_table(data_dir, "cohort", "followup_episodes").merge(
        read_table(data_dir, "outcomes", "episode_labels"), on="episode_id")
    lab = ep[ep["label_status"] == "labelled"].copy()
    lab["year"] = as_utc(lab["prediction_at"]).dt.year
    g = lab.groupby(["year", "programme"])["label"].agg(["size", "mean"]).reset_index()
    tot = lab.groupby("year")["label"].agg(["size", "mean"]).reset_index().assign(programme="all")
    out = pd.concat([g, tot]).sort_values(["year", "programme"])
    return [{"year": int(r.year), "programme": r.programme, "labelled": int(r.size),
             "prevalence": float(r.mean)} for r in out.itertuples()]


def evaluate_bundle(bundle_dir: str | Path, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Reload a saved bundle, rebuild features and the test split from the dataset, and recompute
    the test metrics. They must equal the metrics stored at training time."""
    bundle = load_bundle(bundle_dir)
    md = bundle.metadata
    data_dir = Path(data_dir or md["data_dir"])
    current = data_fingerprint(read_manifest(data_dir))
    feats = build_feature_table(md["cohort_config"], data_dir, write=False)
    table = training_table(md["cohort_config"], md["training_config"], data_dir, feats)
    te = table[table["split"] == "test"].reset_index(drop=True)
    pred = bundle.predict(te)
    X = bundle.matrix(te)
    y = te["label"].to_numpy(int)
    point = M.point_metrics(y, raw_scores(bundle.model, X), pred["probability"].to_numpy(),
                            md["training_config"]["capacity"]["report_fractions"],
                            md["training_config"]["evaluation"]["calibration_bins"])
    stored = json.loads((Path(bundle_dir) / "evaluation.json").read_text(encoding="utf-8"))
    ref = stored["test"]["point"][md["model_type"]]
    keys = ("pr_auc", "roc_auc", "brier", "ece")
    return {"model_version": bundle.version, "model_type": md["model_type"],
            "data_fingerprint_matches": current == md["data_fingerprint"],
            "recomputed": {k: point[k] for k in keys}, "stored": {k: ref[k] for k in keys},
            "matches": all(abs(point[k] - ref[k]) < 1e-12 for k in keys), "n_test": len(te)}


def predict_episodes(bundle_dir: str | Path, data_dir: str | Path | None = None,
                     episode_ids: list[str] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """Score eligible episodes with a saved bundle. Default: episodes still open at the data cutoff."""
    bundle = load_bundle(bundle_dir)
    data_dir = Path(data_dir or bundle.metadata["data_dir"])
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    eligible = episodes[episodes["eligible"]]
    if episode_ids:
        chosen = eligible[eligible["episode_id"].isin(episode_ids)]
        unknown = set(episode_ids) - set(chosen["episode_id"])
        if unknown:
            raise ValueError(f"not eligible or unknown episode ids: {sorted(unknown)}")
    else:
        labels = read_table(data_dir, "outcomes", "episode_labels")
        open_ids = labels.loc[(labels["label_status"] == "immature") & labels["completed_at"].isna(),
                              "episode_id"]
        chosen = eligible[eligible["episode_id"].isin(open_ids)]
    feats = build_features(load_records(data_dir), chosen, bundle.metadata["cohort_config"], episodes)
    return bundle.explain(feats, top_k=top_k)
