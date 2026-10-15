"""Backend test fixtures.

A self-contained environment is built once per session: a small synthetic cohort generated with the
real M1 pipeline, a model bundle trained with the real M2/M3 pipeline (reduced grids), a migrated
SQLite database, the seeded cohort and one scoring run. Nothing depends on the git-ignored
`ml/data/v1` dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from cura_api import migrate
from cura_api.db import make_engine, make_session_factory
from cura_api.main import create_app
from cura_api.services.scoring import run_scoring
from cura_api.services.seed import seed_database
from cura_api.settings import PROJECT_ROOT, Settings
from cura_ml.config import load_config
from cura_ml.modeling.train import train_and_evaluate
from cura_ml.pipeline import run_all

N_PATIENTS = 500
COORD = {"X-Dev-User": "coord.test", "X-Dev-Role": "coordinator"}
ANALYST = {"X-Dev-User": "analyst.test", "X-Dev-Role": "analyst"}
ADMIN = {"X-Dev-User": "admin.test", "X-Dev-Role": "admin"}


@dataclass
class Env:
    root: Path
    data_dir: Path
    bundle_dir: Path
    database_url: str
    settings: Settings
    cohort_cfg: dict
    seed_result: dict
    score_result: dict

    def factory(self):
        engine = make_engine(self.database_url)
        return engine, make_session_factory(engine)


def quick_train_config(version: str = "test-api") -> dict:
    cfg = yaml.safe_load((PROJECT_ROOT / "ml" / "configs" / "train_v1.yaml").read_text("utf-8"))
    cfg["model_version"] = version
    cfg["evaluation"]["bootstrap_resamples"] = 30
    cfg["logistic_regression"]["C_grid"] = [0.03]
    cfg["xgboost"]["max_depth_grid"] = [2]
    cfg["xgboost"]["n_estimators_max"] = 200
    cfg["selection"]["xgboost_pr_auc_ci_lower_bound_above"] = 10.0  # always logistic regression
    return cfg


def make_settings(env_root: Path, database_url: str, data_dir: Path, bundle_dir: Path,
                  **overrides) -> Settings:
    values = {"environment": "test", "auth_mode": "dev", "database_url": database_url,
              "dataset_dir": data_dir, "model_bundle_dir": bundle_dir,
              "model_artifacts_root": bundle_dir.parent, **overrides}
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def env(tmp_path_factory) -> Env:
    root = tmp_path_factory.mktemp("backend")
    cohort_cfg = load_config(PROJECT_ROOT / "ml" / "configs" / "cohort_v1.yaml",
                             {"simulation": {"n_patients": N_PATIENTS}})
    data_dir = root / "data"
    run_all(cohort_cfg, data_dir)
    res = train_and_evaluate(cohort_cfg, quick_train_config(), data_dir, root / "models", None)
    url = f"sqlite:///{(root / 'cura_test.db').as_posix()}"
    migrate.upgrade(url)
    engine = make_engine(url)
    factory = make_session_factory(engine)
    seed_result = seed_database(factory, data_dir, actor="system:test")
    score_result = run_scoring(factory, res["bundle_dir"], data_dir, triggered_by="system:test")
    engine.dispose()
    settings = make_settings(root, url, data_dir, Path(res["bundle_dir"]))
    return Env(root, data_dir, Path(res["bundle_dir"]), url, settings, cohort_cfg, seed_result,
               score_result)


@pytest.fixture(scope="session")
def client(env) -> TestClient:
    with TestClient(create_app(env.settings)) as c:
        yield c
