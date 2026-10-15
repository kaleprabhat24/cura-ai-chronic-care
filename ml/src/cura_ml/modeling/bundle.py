"""Inference-ready model bundle: save, load, predict and explain.

Layout of `ml/artifacts/models/<model_version>/`:
    model.joblib          fitted model (incl. preprocessing pipeline) + calibrator
    metadata.json         version, model type, feature columns, thresholds, data hash, configs, libraries
    feature_spec.json     the feature specification the model was trained with
    evaluation.json       validation and test results
    model_card.md         intended use, data, results and limitations
    xgboost_model.json    portable XGBoost model (only when XGBoost is selected)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from cura_ml.features import spec as S
from cura_ml.modeling.explain import contributions, explain_rows
from cura_ml.modeling.models import Calibrator, raw_log_odds, raw_scores

TIERS = ("high", "medium", "low")


def assign_tiers(prob: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return np.where(prob >= thresholds["high"], "high",
                    np.where(prob >= thresholds["medium"], "medium", "low"))


@dataclass
class ModelBundle:
    model: Any
    calibrator: Calibrator
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def feature_columns(self) -> list[str]:
        return self.metadata["feature_columns"]

    def matrix(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"feature frame lacks columns: {missing[:5]}")
        return features[self.feature_columns].to_numpy(dtype=float)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = self.matrix(features)
        prob = self.calibrator.transform(raw_scores(self.model, X))
        return pd.DataFrame({
            "episode_id": features["episode_id"].to_numpy(),
            "model_log_odds": raw_log_odds(self.model, X),
            "probability": prob,
            "risk_tier": assign_tiers(prob, self.metadata["tier_thresholds"]),
            "model_version": self.version,
            "provenance": "simulated_prediction",
        })

    def explain(self, features: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
        X = self.matrix(features)
        contrib, base, method = contributions(self.model, X)
        rows = explain_rows(contrib, base, X, self.feature_columns,
                            list(features["episode_id"]), method, top_k)
        pred = self.predict(features)
        for row, p, tier in zip(rows, pred["probability"], pred["risk_tier"], strict=True):
            row["calibrated_probability"] = round(float(p), 4)
            row["risk_tier"] = tier
            row["model_version"] = self.version
        return rows


def save_bundle(path: str | Path, model: Any, calibrator: Calibrator, metadata: dict[str, Any],
                evaluation: dict[str, Any], model_card: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator}, path / "model.joblib")
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(path / "xgboost_model.json")
    for name, obj in (("metadata.json", metadata), ("evaluation.json", evaluation),
                      ("feature_spec.json", {"version": S.FEATURE_SPEC_VERSION,
                                             "columns": S.feature_columns(),
                                             "features": S.spec_records()})):
        (path / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json) + "\n",
                                 encoding="utf-8")
    (path / "model_card.md").write_text(model_card, encoding="utf-8")
    return path


def load_bundle(path: str | Path) -> ModelBundle:
    path = Path(path)
    blob = joblib.load(path / "model.joblib")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_spec_version"] != S.FEATURE_SPEC_VERSION or \
            metadata["feature_columns"] != S.feature_columns():
        raise ValueError("bundle was trained with a different feature specification than the code")
    return ModelBundle(blob["model"], blob["calibrator"], metadata)


def _json(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
