"""Batch scoring with the saved model bundle.

* Uses the bundle produced by `cura-ml train` (model, preprocessing, calibrator, tier thresholds) and
  the existing `cura_ml.features.build_features`, configured with the bundle's own cohort config.
  Every feature is therefore computed from records recorded before each episode's prediction time.
* Refuses to score when the database, the dataset files and the bundle do not refer to the same data
  (fingerprint check), unless explicitly overridden.
* Each run is a new, immutable snapshot (`scoring_runs` + `risk_scores`); earlier runs are kept.
  Nothing here is called from API read endpoints — scores are never recomputed on page load.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cura_api import models as m
from cura_api.db import utcnow
from cura_api.services import audit
from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import load_records, read_table
from cura_ml.features import spec as S
from cura_ml.features.build import build_features
from cura_ml.manifest import data_fingerprint, read_manifest
from cura_ml.modeling.bundle import load_bundle

Scope = Literal["eligible", "open"]


class ScoringError(RuntimeError):
    pass


def bundle_metadata_sha256(bundle_dir: str | Path) -> str:
    return hashlib.sha256((Path(bundle_dir) / "metadata.json").read_bytes()).hexdigest()


def select_episodes(data_dir: Path, scope: Scope) -> pd.DataFrame:
    """Eligible episodes (t0 within the study period and on/before the cutoff). With scope "open":
    only those still awaiting completion at the cutoff (not completed, not censored, patient not
    known to have died or transferred out)."""
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    eligible = episodes[episodes["eligible"]]
    if scope == "eligible":
        return eligible
    labels = read_table(data_dir, "outcomes", "episode_labels").set_index("episode_id")
    pts = read_table(data_dir, "records", "patients").set_index("patient_id")
    lab = labels.loc[eligible["episode_id"]]
    exited = pts.loc[eligible["patient_id"], ["death_date", "transfer_out_date"]].notna().any(axis=1)
    keep = (lab["completed_at"].isna().to_numpy() & ~lab["censored"].to_numpy().astype(bool)
            & ~exited.to_numpy())
    return eligible[keep]


def _snapshot(row: pd.Series, columns: list[str]) -> dict[str, float | None]:
    out = {}
    for c in columns:
        v = float(row[c])
        out[c] = None if math.isnan(v) else v
    return out


def run_scoring(factory: sessionmaker[Session], bundle_dir: str | Path, data_dir: str | Path,
                scope: Scope = "eligible", triggered_by: str = "system:cli", actor_role: str = "system",
                allow_fingerprint_mismatch: bool = False) -> dict[str, Any]:
    bundle_dir, data_dir = Path(bundle_dir), Path(data_dir)
    bundle = load_bundle(bundle_dir)
    md = bundle.metadata
    manifest = read_manifest(data_dir)
    fingerprint = data_fingerprint(manifest)
    started = utcnow()

    with factory() as session:
        load = session.execute(select(m.DatasetLoad).order_by(m.DatasetLoad.id.desc())).scalars().first()
    problems = []
    if load is None:
        raise ScoringError("the database has no loaded cohort; run `cura-api seed` first")
    if load.data_fingerprint != fingerprint:
        problems.append("database was loaded from different dataset files than the current ones")
    if md["data_fingerprint"] != fingerprint:
        problems.append("model bundle was trained on a different dataset fingerprint")
    if problems and not allow_fingerprint_mismatch:
        raise ScoringError("; ".join(problems) + " (re-seed / retrain, or pass --allow-fingerprint-mismatch)")

    target = select_episodes(data_dir, scope)
    episodes_all = read_table(data_dir, "cohort", "followup_episodes")
    feats = build_features(load_records(data_dir), target, md["cohort_config"], episodes_all)
    pred = bundle.predict(feats)
    explanations = bundle.explain(feats, top_k=5)
    order = np.lexsort((pred["episode_id"].to_numpy(), -pred["model_log_odds"].to_numpy(),
                        -pred["probability"].to_numpy()))
    rank = np.empty(len(pred), dtype=int)
    rank[order] = np.arange(1, len(pred) + 1)
    scored_at = utcnow()
    prediction_at = as_utc(feats["prediction_at"])
    cols = bundle.feature_columns

    with factory() as session, session.begin():
        run = m.ScoringRun(
            status="succeeded", scope=scope, model_version=bundle.version, model_type=md["model_type"],
            bundle_path=str(bundle_dir), bundle_metadata_sha256=bundle_metadata_sha256(bundle_dir),
            feature_spec_version=md["feature_spec_version"], calibration_method=md["calibration_method"],
            tier_thresholds=md["tier_thresholds"], data_fingerprint=fingerprint,
            data_cutoff=date.fromisoformat(manifest["data_cutoff"]), started_at=started,
            finished_at=None, n_scored=len(pred),
            tier_counts=dict(Counter(pred["risk_tier"].tolist())), triggered_by=triggered_by,
            error=None if not problems else "; ".join(problems))
        session.add(run)
        session.flush()
        rows = []
        for i, (frow, p, e) in enumerate(zip(feats.itertuples(index=False), pred.itertuples(index=False),
                                             explanations, strict=True)):
            rows.append({
                "scoring_run_id": run.id, "episode_id": p.episode_id, "patient_id": frow.patient_id,
                "model_version": bundle.version, "probability": float(p.probability),
                "model_log_odds": float(p.model_log_odds), "risk_tier": p.risk_tier,
                "rank": int(rank[i]), "prediction_at": prediction_at.iloc[i].to_pydatetime(),
                "scored_at": scored_at, "explanation": e,
                "feature_snapshot": _snapshot(feats.iloc[i], cols),
            })
        for j in range(0, len(rows), 2000):
            session.execute(m.RiskScore.__table__.insert(), rows[j:j + 2000])
        run.finished_at = utcnow()
        audit.record(session, triggered_by, actor_role, "scoring.run", "scoring_run", str(run.id),
                     {"model_version": bundle.version, "scope": scope, "n_scored": len(rows),
                      "data_fingerprint": fingerprint, "tier_counts": run.tier_counts,
                      "fingerprint_override": bool(problems)})
        run_id = run.id
        tier_counts = run.tier_counts
    return {"scoring_run_id": run_id, "model_version": bundle.version, "model_type": md["model_type"],
            "scope": scope, "n_scored": len(rows), "tier_counts": tier_counts,
            "tier_thresholds": md["tier_thresholds"], "data_fingerprint": fingerprint,
            "feature_spec_version": S.FEATURE_SPEC_VERSION}
