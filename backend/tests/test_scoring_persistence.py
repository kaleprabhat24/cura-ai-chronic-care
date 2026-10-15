"""Model-bundle loading, batch scoring persistence and snapshot preservation."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from sqlalchemy import func, select

from cura_api import models as m
from cura_api.db import make_engine, make_session_factory
from cura_api.services.scoring import ScoringError, bundle_metadata_sha256, run_scoring
from cura_ml.dataset import load_records, read_table
from cura_ml.features import spec as S
from cura_ml.features.build import build_features
from cura_ml.modeling.bundle import load_bundle


def test_bundle_loads_and_matches_dataset(env):
    bundle = load_bundle(env.bundle_dir)
    assert bundle.metadata["model_type"] == "logistic_regression"
    assert bundle.feature_columns == S.feature_columns()
    assert env.score_result["tier_thresholds"] == bundle.metadata["tier_thresholds"]


def test_scores_persisted_with_provenance(env):
    engine, factory = env.factory()
    bundle = load_bundle(env.bundle_dir)
    with factory() as s:
        run = s.get(m.ScoringRun, env.score_result["scoring_run_id"])
        assert run.status == "succeeded" and run.scope == "eligible"
        assert run.model_version == bundle.version and run.model_type == "logistic_regression"
        assert run.bundle_metadata_sha256 == bundle_metadata_sha256(env.bundle_dir)
        assert run.tier_thresholds == bundle.metadata["tier_thresholds"]
        assert run.data_fingerprint == bundle.metadata["data_fingerprint"]
        assert run.feature_spec_version == S.FEATURE_SPEC_VERSION
        n_eligible = s.execute(select(func.count()).select_from(m.FollowupEpisode).where(
            m.FollowupEpisode.eligible.is_(True))).scalar_one()
        scores = s.execute(select(m.RiskScore).where(m.RiskScore.scoring_run_id == run.id)).scalars().all()
        assert len(scores) == n_eligible == run.n_scored
        assert sorted(r.rank for r in scores) == list(range(1, len(scores) + 1))
        events = s.execute(select(m.AuditEvent).where(m.AuditEvent.action == "scoring.run")).scalars().all()
        assert any(e.entity_id == str(run.id) for e in events)
    # probabilities, tiers and feature snapshots equal a fresh computation with the same bundle
    episodes = read_table(env.data_dir, "cohort", "followup_episodes")
    sample = episodes[episodes["eligible"]].sample(40, random_state=1)
    feats = build_features(load_records(env.data_dir), sample, bundle.metadata["cohort_config"], episodes)
    pred = bundle.predict(feats).set_index("episode_id")
    with factory() as s:
        for r in s.execute(select(m.RiskScore).where(m.RiskScore.scoring_run_id == run.id,
                                                      m.RiskScore.episode_id.in_(list(sample["episode_id"])))).scalars():
            assert r.probability == pytest.approx(pred.loc[r.episode_id, "probability"], abs=1e-12)
            assert r.risk_tier == pred.loc[r.episode_id, "risk_tier"]
            expected = feats.set_index("episode_id").loc[r.episode_id, S.feature_columns()].to_numpy(float)
            got = np.array([np.nan if r.feature_snapshot[c] is None else r.feature_snapshot[c]
                            for c in S.feature_columns()])
            np.testing.assert_allclose(got, expected, equal_nan=True)
            assert r.explanation["factors"] and r.explanation["method"] == "logistic_regression_contributions"
            th = run.tier_thresholds
            want = ("high" if r.probability >= th["high"]
                    else "medium" if r.probability >= th["medium"] else "low")
            assert r.risk_tier == want
    engine.dispose()


def test_new_run_preserves_previous_snapshots(env, tmp_path):
    db = tmp_path / "copy.db"
    shutil.copy(env.root / "cura_test.db", db)
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    factory = make_session_factory(engine)
    with factory() as s:
        before = s.execute(select(m.RiskScore.id, m.RiskScore.probability)).all()
    second = run_scoring(factory, env.bundle_dir, env.data_dir, scope="open", triggered_by="system:test")
    with factory() as s:
        assert s.execute(select(func.count()).select_from(m.ScoringRun)).scalar_one() == 2
        kept = dict(s.execute(select(m.RiskScore.id, m.RiskScore.probability).where(
            m.RiskScore.scoring_run_id == env.score_result["scoring_run_id"])).all())
        assert kept == dict(before)
        new = s.execute(select(m.RiskScore).where(
            m.RiskScore.scoring_run_id == second["scoring_run_id"])).scalars().all()
        assert len(new) == second["n_scored"] > 0
        # same model + same data -> same probability for the same episode in both snapshots
        for r in new[:50]:
            assert r.probability == pytest.approx(kept[s.execute(select(m.RiskScore.id).where(
                m.RiskScore.scoring_run_id == env.score_result["scoring_run_id"],
                m.RiskScore.episode_id == r.episode_id)).scalar_one()], abs=1e-12)
    engine.dispose()


def test_scoring_refuses_mismatched_data(env, tmp_path):
    db = tmp_path / "copy.db"
    shutil.copy(env.root / "cura_test.db", db)
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    factory = make_session_factory(engine)
    bundle = tmp_path / "bundle"
    shutil.copytree(env.bundle_dir, bundle)
    md = json.loads((bundle / "metadata.json").read_text("utf-8"))
    md["data_fingerprint"] = "0" * 64
    (bundle / "metadata.json").write_text(json.dumps(md), encoding="utf-8")
    with pytest.raises(ScoringError, match="different dataset fingerprint"):
        run_scoring(factory, bundle, env.data_dir)
    with factory() as s:
        assert s.execute(select(func.count()).select_from(m.ScoringRun)).scalar_one() == 1
    engine.dispose()


def test_reading_api_never_creates_scores(env, client):
    from conftest import COORD

    engine, factory = env.factory()
    with factory() as s:
        runs = s.execute(select(func.count()).select_from(m.ScoringRun)).scalar_one()
        scores = s.execute(select(func.count()).select_from(m.RiskScore)).scalar_one()
    item = client.get("/api/v1/worklist?page_size=1", headers=COORD).json()["items"][0]
    client.get(f"/api/v1/patients/{item['patient']['id']}", headers=COORD)
    client.get(f"/api/v1/risk-scores/{item['prediction']['risk_score_id']}", headers=COORD)
    client.get("/api/v1/dashboard/summary", headers=COORD)
    with factory() as s:
        assert s.execute(select(func.count()).select_from(m.ScoringRun)).scalar_one() == runs
        assert s.execute(select(func.count()).select_from(m.RiskScore)).scalar_one() == scores
    engine.dispose()
