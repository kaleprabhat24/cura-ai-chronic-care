from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cura_ml.config import load_config
from cura_ml.pipeline import run_all

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cohort_v1.yaml"
TEST_PATIENTS = 2000


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory) -> tuple[dict, Path]:
    """A generated cohort shared by the dataset-level tests (same config, fewer patients)."""
    cfg = load_config(CONFIG_PATH, {"simulation": {"n_patients": TEST_PATIENTS}})
    out = tmp_path_factory.mktemp("cohort")
    run_all(cfg, out)
    return cfg, out


@pytest.fixture(scope="session")
def cutoff_end(base_cfg) -> pd.Timestamp:
    """Exclusive end of the data cutoff day."""
    return pd.Timestamp(base_cfg["simulation"]["data_cutoff"], tz="UTC") + pd.Timedelta(days=1)
