"""Reproducibility: same seed + config -> identical data; different seed -> different data."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from cura_ml.config import ConfigError, config_hash, load_config
from cura_ml.pipeline import run_all
from cura_ml.synth.generator import generate

from .conftest import CONFIG_PATH


def _small(n: int = 150, **sim):
    return load_config(CONFIG_PATH, {"simulation": {"n_patients": n, **sim}})


def test_same_seed_gives_identical_content_and_file_hashes(tmp_path):
    cfg = _small()
    m1 = run_all(cfg, tmp_path / "a")
    m2 = run_all(cfg, tmp_path / "b")
    assert m1["config_sha256"] == m2["config_sha256"]
    assert set(m1["files"]) == set(m2["files"])
    for name in m1["files"]:
        assert m1["files"][name]["content_sha256"] == m2["files"][name]["content_sha256"], name
        assert m1["files"][name]["file_sha256"] == m2["files"][name]["file_sha256"], name


def test_different_seed_gives_different_records(tmp_path):
    cfg = _small()
    other = copy.deepcopy(cfg)
    other["seed"] = cfg["seed"] + 1
    m1 = run_all(cfg, tmp_path / "a")
    m2 = run_all(other, tmp_path / "b")
    assert m1["config_sha256"] != m2["config_sha256"]
    for table in ("patients", "appointments", "observations"):
        key = f"records/{table}.parquet"
        assert m1["files"][key]["content_sha256"] != m2["files"][key]["content_sha256"]


def test_existing_patients_unchanged_when_cohort_grows():
    small, _ = generate(_small(40))
    large, _ = generate(_small(60))
    for table in ("patients", "appointments", "encounters", "observations"):
        a = getattr(small, table)
        b = getattr(large, table)
        b = b[b["patient_id"].isin(a["patient_id"].unique())].reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b)


def test_config_hash_is_canonical_and_sensitive():
    cfg = _small()
    reordered = dict(reversed(list(cfg.items())))
    assert config_hash(cfg) == config_hash(reordered)
    changed = copy.deepcopy(cfg)
    changed["attendance"]["base_logit"] += 0.01
    assert config_hash(cfg) != config_hash(changed)


def test_config_rejects_unknown_and_missing_keys():
    with pytest.raises(ConfigError, match="unknown"):
        load_config(CONFIG_PATH, {"attendance": {"base_logti": 1.0}})
    cfg = load_config(CONFIG_PATH)
    del cfg["episodes"]["prediction_lead_days"]
    from cura_ml.config import validate_config

    with pytest.raises(ConfigError, match="missing"):
        validate_config(cfg)
