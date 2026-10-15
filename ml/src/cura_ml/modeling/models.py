"""Model definitions: two baselines, logistic regression, XGBoost, and probability calibration.

Every model exposes raw scores via `raw_scores(model, X)`. Ranking metrics and the worklist order
use raw scores. Probabilities shown to users come from a `Calibrator` fitted on validation data only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

EPS = 1e-6


class PrevalenceBaseline:
    """Predicts the training-set prevalence for everyone (no ranking ability)."""

    model_type = "prevalence_baseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PrevalenceBaseline:
        self.p_ = float(np.mean(y))
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.p_)


class AttendanceHeuristic:
    """Rule of thumb: risk score = 1 − smoothed 24-month attendance rate."""

    model_type = "attendance_heuristic"

    def __init__(self, feature_index: int) -> None:
        self.feature_index = feature_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> AttendanceHeuristic:
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - X[:, self.feature_index]


def make_logistic_regression(C: float) -> Pipeline:
    """Median imputation + standardisation + L2 logistic regression. Every step is fitted on the
    training data passed to `.fit` only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000)),
    ])


def make_xgboost(cfg: dict[str, Any], max_depth: int, n_estimators: int, seed: int,
                 early_stopping_rounds: int | None = None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"], reg_lambda=cfg["reg_lambda"],
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=cfg["n_jobs"], random_state=seed, early_stopping_rounds=early_stopping_rounds,
    )


def raw_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "raw_scores"):
        return model.raw_scores(X)
    return model.predict_proba(X)[:, 1]


def raw_log_odds(model: Any, X: np.ndarray) -> np.ndarray:
    """Uncalibrated log-odds of a fitted LR pipeline or XGBoost model (explanations add up to this)."""
    if isinstance(model, Pipeline):
        return model.decision_function(X)
    return model.predict(X, output_margin=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw scores to calibrated probabilities ('platt' = logistic on logit, or 'isotonic')."""

    def __init__(self, method: str) -> None:
        if method not in ("platt", "isotonic", "identity"):
            raise ValueError(method)
        self.method = method

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Calibrator:
        self.n_fit_ = len(y)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores)[:, None], y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.model_.predict_proba(_logit(scores)[:, None])[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model_.predict(scores), 0.0, 1.0)
        return np.asarray(scores, dtype=float)


def choose_calibrator(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, methods: list[str],
                      folds: int) -> tuple[Calibrator, dict[str, float]]:
    """Pick the calibration method with the lowest patient-grouped cross-validated Brier score on
    the calibration data, then refit it on all of that data."""
    cv: dict[str, float] = {}
    for method in methods:
        losses = []
        for fit_idx, eval_idx in GroupKFold(n_splits=folds).split(scores, y, groups):
            cal = Calibrator(method).fit(scores[fit_idx], y[fit_idx])
            losses.append(np.mean((cal.transform(scores[eval_idx]) - y[eval_idx]) ** 2))
        cv[method] = float(np.mean(losses))
    best = min(methods, key=lambda m: (cv[m], methods.index(m)))
    return Calibrator(best).fit(scores, y), cv
