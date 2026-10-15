"""Evaluation metrics with patient-level bootstrap confidence intervals.

Ranking metrics (PR-AUC, ROC-AUC, precision/recall at capacity) use raw model scores, which give
the worklist order. Probability metrics (Brier, calibration) use calibrated probabilities.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def _order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Descending score order with a seeded random tie-break (matters for constant scores)."""
    tiebreak = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tiebreak, -score))


def capacity_metrics(y: np.ndarray, score: np.ndarray, fraction: float) -> dict[str, float]:
    k = max(1, math.ceil(fraction * len(y)))
    top = _order(score)[:k]
    precision = float(y[top].mean())
    recall = float(y[top].sum() / max(y.sum(), 1))
    prevalence = float(y.mean())
    return {"k": k, "precision": precision, "recall": recall,
            "lift": precision / prevalence if prevalence else float("nan")}


def calibration_stats(y: np.ndarray, prob: np.ndarray, bins: int) -> dict[str, float]:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], prob, side="right"), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(prob[sel].mean() - y[sel].mean())
    out = {"ece": float(ece), "mean_predicted": float(prob.mean()), "observed_rate": float(y.mean()),
           "calibration_in_the_large": float(y.mean() - prob.mean())}
    if np.ptp(prob) > 1e-9:
        logit = np.log(np.clip(prob, EPS, 1 - EPS) / (1 - np.clip(prob, EPS, 1 - EPS)))
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
        out["calibration_slope"] = float(lr.coef_[0, 0])
        out["calibration_intercept"] = float(lr.intercept_[0])
    else:
        out["calibration_slope"] = out["calibration_intercept"] = float("nan")
    return out


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int) -> list[dict[str, float]]:
    order = np.argsort(prob, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            rows.append({"n": int(len(chunk)), "mean_predicted": float(prob[chunk].mean()),
                         "observed_rate": float(y[chunk].mean()),
                         "min_predicted": float(prob[chunk].min()),
                         "max_predicted": float(prob[chunk].max())})
    return rows


def point_metrics(y: np.ndarray, score: np.ndarray, prob: np.ndarray, fractions: list[float],
                  bins: int) -> dict[str, object]:
    out: dict[str, object] = {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(np.mean((prob - y) ** 2)),
    }
    out.update(calibration_stats(y, prob, bins))
    out["capacity"] = {f"{fr:.2f}": capacity_metrics(y, score, fr) for fr in fractions}
    return out


def _core(y: np.ndarray, score: np.ndarray, prob: np.ndarray, fraction: float) -> dict[str, float]:
    cap = capacity_metrics(y, score, fraction)
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "brier": float(np.mean((prob - y) ** 2)),
            "precision_at_capacity": cap["precision"], "recall_at_capacity": cap["recall"]}


def bootstrap(y: np.ndarray, groups: np.ndarray, models: dict[str, tuple[np.ndarray, np.ndarray]],
              fraction: float, n_resamples: int, seed: int,
              pairs: list[tuple[str, str]]) -> dict[str, object]:
    """Patient-level bootstrap: resample patients with replacement and keep all their episodes.
    Returns 95% percentile intervals per model and for paired differences (a − b)."""
    rng = np.random.default_rng(seed)
    patients, inverse = np.unique(groups, return_inverse=True)
    rows_by_patient = [np.flatnonzero(inverse == i) for i in range(len(patients))]
    samples: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([rows_by_patient[i] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for name, (score, prob) in models.items():
            samples[name].append(_core(yb, score[idx], prob[idx], fraction))

    def interval(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    out: dict[str, object] = {"n_resamples_used": len(next(iter(samples.values()))), "models": {},
                              "differences": {}}
    for name, rows in samples.items():
        out["models"][name] = {k: interval([r[k] for r in rows]) for k in rows[0]}
    for a, b in pairs:
        diffs = {}
        for k in ("pr_auc", "roc_auc", "brier", "precision_at_capacity"):
            d = [ra[k] - rb[k] for ra, rb in zip(samples[a], samples[b], strict=True)]
            diffs[k] = {"interval": interval(d), "share_positive": float(np.mean(np.array(d) > 0))}
        out["differences"][f"{a} - {b}"] = diffs
    return out


def single_feature_auc(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Discrimination of each feature on its own (max(AUC, 1 − AUC)), NaN filled with the median."""
    out = []
    for j, name in enumerate(names):
        col = X[:, j].astype(float)
        if np.all(np.isnan(col)) or np.nanstd(col) == 0:
            out.append((name, 0.5))
            continue
        col = np.where(np.isnan(col), np.nanmedian(col), col)
        auc = roc_auc_score(y, col)
        out.append((name, float(max(auc, 1 - auc))))
    return sorted(out, key=lambda t: -t[1])
