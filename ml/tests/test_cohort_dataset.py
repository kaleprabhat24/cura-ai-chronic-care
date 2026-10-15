"""Cohort-level acceptance checks: prevalence band, manifest integrity and report generation."""

from __future__ import annotations

import pandas as pd

from cura_ml.cohort.report import build_report
from cura_ml.config import config_hash
from cura_ml.dataset import LAYOUT, content_hash, file_hash, read_table
from cura_ml.manifest import read_manifest


def test_label_prevalence_within_configured_band(small_dataset):
    cfg, data_dir = small_dataset
    labels = read_table(data_dir, "outcomes", "episode_labels")
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    lab = labels[labels["label_status"] == "labelled"]
    assert len(lab) >= 1000
    lo, hi = cfg["checks"]["label_prevalence_band"]
    assert lo <= lab["label"].astype(float).mean() <= hi
    by_prog = episodes.merge(lab, on="episode_id").groupby("programme")["label"].mean()
    assert set(by_prog.index) == set(cfg["programmes"])
    assert ((by_prog > 0) & (by_prog < 1)).all()


def test_label_only_on_eligible_mature_uncensored(small_dataset):
    _, data_dir = small_dataset
    ep = read_table(data_dir, "cohort", "followup_episodes").merge(
        read_table(data_dir, "outcomes", "episode_labels"), on="episode_id")
    has_label = ep["label"].notna()
    assert (has_label == (ep["label_status"] == "labelled")).all()
    assert ep.loc[has_label, "eligible"].all()
    assert ep.loc[has_label, "mature"].all()
    assert not ep.loc[has_label, "censored"].any()
    assert (ep["eligible"] == ep["ineligible_reason"].isna()).all()
    assert {"immature", "censored", "ineligible", "labelled"} <= set(ep["label_status"])


def test_manifest_matches_files(small_dataset):
    cfg, data_dir = small_dataset
    manifest = read_manifest(data_dir)
    assert manifest["seed"] == cfg["seed"]
    assert manifest["config_sha256"] == config_hash(cfg)
    assert manifest["synthetic_data_only"] is True
    expected = {f"{g}/{t}.parquet" for g, tables in LAYOUT.items() for t in tables}
    assert set(manifest["files"]) == expected
    for name, info in manifest["files"].items():
        path = data_dir / name
        df = pd.read_parquet(path)
        assert info["rows"] == len(df)
        assert info["content_sha256"] == content_hash(df)
        assert info["file_sha256"] == file_hash(path)


def test_report_builds_and_reports_key_sections(small_dataset):
    cfg, data_dir = small_dataset
    text = build_report(data_dir, cfg)
    for heading in ("## Patients", "## Follow-up episodes", "## Label prevalence",
                    "## Outcome maturity", "## Missingness", "Information available at prediction"):
        assert heading in text
    assert "PASS" in text
