"""Per-prediction explanations.

* Logistic regression: exact additive contributions coef_j × standardised value_j. They sum, with the
  intercept, to the model's log-odds.
* XGBoost: TreeSHAP values from the `shap` library, also on the log-odds scale and summing, with the
  expected value, to the model's raw margin.

Contributions are in **log-odds of the uncalibrated model**. They are *not* percentage points of risk:
the displayed probability comes from a separate calibration step. A feature and its `__missing`
indicator are reported together as one factor. Contributions describe how the model used the
recorded data. They are associations, not causes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.pipeline import Pipeline

from cura_ml.features import spec as S

UNITS_NOTE = ("Contributions are log-odds on the uncalibrated model scale (positive raises, negative "
              "lowers the model's risk estimate). They are not percentages of risk probability.")
CAUSAL_NOTE = ("Explanations show which recorded data the model relied on for this prediction. They are "
               "statistical associations in synthetic data, not causes, and not clinical advice.")


def contributions(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """(n × p contributions, n base values, method name) on the log-odds scale."""
    if isinstance(model, Pipeline):
        imp, sc, clf = (model.named_steps[k] for k in ("impute", "scale", "clf"))
        z = sc.transform(imp.transform(X))
        contrib = z * clf.coef_[0]
        return contrib, np.full(len(X), float(clf.intercept_[0])), "logistic_regression_contributions"
    if isinstance(model, xgb.XGBClassifier):
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X), dtype=float)
        base = float(np.ravel(explainer.expected_value)[0])
        return values, np.full(len(X), base), "tree_shap"
    raise TypeError(f"no explanation method for {type(model).__name__}")


def native_tree_contributions(model: xgb.XGBClassifier, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost's built-in TreeSHAP (`pred_contribs`), used to cross-check the shap library."""
    out = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return out[:, :-1], out[:, -1]


def grouped(contrib: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum each feature's contribution with its missing-indicator's contribution."""
    df = pd.DataFrame(contrib, columns=columns)
    return df.T.groupby(lambda c: S.base_feature(c), sort=False).sum().T


def _display_value(name: str, value: float) -> Any:
    if np.isnan(value):
        return None
    unit = S.FEATURE_BY_NAME[name].unit
    if unit == "yes/no":
        return "yes" if value >= 0.5 else "no"
    return round(float(value), 2)


def _value_text(name: str, value: float) -> str:
    f = S.FEATURE_BY_NAME[name]
    shown = _display_value(name, value)
    if shown is None:
        return f.missing_label
    if f.unit in ("count", "quintile"):
        return str(int(round(shown)))
    if f.unit in ("yes/no", "ratio"):
        return str(shown)
    return f"{shown} {f.unit}"


def explain_rows(contrib: np.ndarray, base: np.ndarray, X: np.ndarray, columns: list[str],
                 episode_ids: list[str], method: str, top_k: int = 5) -> list[dict[str, Any]]:
    g = grouped(contrib, columns)
    col_index = {c: i for i, c in enumerate(columns)}
    out = []
    for r, episode_id in enumerate(episode_ids):
        row = g.iloc[r]
        order = row.abs().sort_values(ascending=False, kind="stable").index[:top_k]
        factors = []
        for name in order:
            value = X[r, col_index[name]]
            c = float(row[name])
            factors.append({
                "feature": name,
                "label": S.FEATURE_BY_NAME[name].label,
                "value": _display_value(name, value),
                "value_text": _value_text(name, value),
                "unit": S.FEATURE_BY_NAME[name].unit,
                "recorded": not np.isnan(value),
                "contribution_log_odds": round(c, 4),
                "direction": "raises the model's risk estimate" if c > 0
                else "lowers the model's risk estimate",
            })
        out.append({
            "episode_id": episode_id, "method": method,
            "base_log_odds": round(float(base[r]), 4),
            "model_log_odds": round(float(base[r] + contrib[r].sum()), 4),
            "factors": factors, "units": UNITS_NOTE, "note": CAUSAL_NOTE,
        })
    return out


def global_importance(contrib: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    g = grouped(contrib, columns).abs().mean().sort_values(ascending=False)
    return [{"feature": k, "label": S.FEATURE_BY_NAME[k].label, "mean_abs_log_odds": float(v)}
            for k, v in g.items()]
