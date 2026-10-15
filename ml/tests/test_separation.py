"""Hidden traits, future outcomes and ordinary records stay separate."""

from __future__ import annotations

import shutil
from dataclasses import fields
from pathlib import Path

import pandas as pd

from cura_ml.cohort.episodes import as_utc, build_episodes
from cura_ml.dataset import (
    COHORT_TABLES,
    HIDDEN_TABLES,
    OUTCOME_TABLES,
    RECORD_TABLES,
    Records,
    load_records,
    read_table,
    write_table,
)

OUTCOME_COLUMNS = {
    "completed_at", "completion_event_type", "completion_event_ref", "completed_in_window",
    "censored", "censor_reason", "censor_date", "mature", "label_status", "label",
}


def _columns(data_dir: Path, group: str, tables) -> dict[str, set[str]]:
    return {t: set(read_table(data_dir, group, t).columns) for t in tables}


def test_directory_layout(small_dataset):
    _, data_dir = small_dataset
    for group, tables in (("records", RECORD_TABLES), ("hidden", HIDDEN_TABLES),
                          ("cohort", COHORT_TABLES), ("outcomes", OUTCOME_TABLES)):
        found = {p.stem for p in (data_dir / group).glob("*.parquet")}
        assert found == set(tables), group


def test_records_type_cannot_carry_hidden_or_outcome_tables():
    assert tuple(f.name for f in fields(Records)) == RECORD_TABLES
    assert not set(RECORD_TABLES) & (set(HIDDEN_TABLES) | set(OUTCOME_TABLES))


def test_hidden_trait_columns_absent_from_ordinary_data(small_dataset):
    _, data_dir = small_dataset
    hidden_cols = set(read_table(data_dir, "hidden", "simulation_traits").columns) - {"patient_id"}
    ordinary = {**_columns(data_dir, "records", RECORD_TABLES),
                **_columns(data_dir, "cohort", COHORT_TABLES),
                **_columns(data_dir, "outcomes", OUTCOME_TABLES)}
    for table, cols in ordinary.items():
        assert not cols & hidden_cols, f"hidden trait column in {table}"


def test_episode_definitions_contain_no_outcome_columns(small_dataset):
    _, data_dir = small_dataset
    cols = set(read_table(data_dir, "cohort", "followup_episodes").columns)
    assert not cols & OUTCOME_COLUMNS
    assert set(read_table(data_dir, "outcomes", "episode_labels").columns) >= OUTCOME_COLUMNS


def test_episodes_rebuild_from_records_only(small_dataset, tmp_path):
    """Copy only records/ elsewhere and rebuild: results must equal the saved cohort/outcomes."""
    cfg, data_dir = small_dataset
    shutil.copytree(data_dir / "records", tmp_path / "records")
    episodes, labels = build_episodes(load_records(tmp_path), cfg)
    assert not (tmp_path / "hidden").exists()
    for df, group, table in ((episodes, "cohort", "followup_episodes"),
                             (labels, "outcomes", "episode_labels")):
        write_table(df, tmp_path, group, table)  # same Parquet round trip as the saved copy
        pd.testing.assert_frame_equal(read_table(tmp_path, group, table),
                                      read_table(data_dir, group, table))


def test_builder_agrees_with_simulator_ground_truth(small_dataset):
    """The simulator's private episode log and the record-based reconstruction must match."""
    _, data_dir = small_dataset
    log = read_table(data_dir, "hidden", "simulator_episode_log")
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    labels = read_table(data_dir, "outcomes", "episode_labels")
    built = episodes.merge(labels, on="episode_id")
    built["index_date"] = as_utc(built["index_event_at"]).dt.date
    built["completion_date"] = as_utc(built["completed_at"]).dt.date
    key = ["patient_id", "programme", "index_date"]
    m = log.assign(index_date=pd.to_datetime(log["index_date"]).dt.date).merge(
        built[key + ["completion_date", "due_date"]], on=key, how="outer",
        suffixes=("_sim", "_built"), indicator=True)
    assert (m["_merge"] == "both").all(), m[m["_merge"] != "both"].head()
    sim_c = pd.to_datetime(m["completion_date_sim"])
    built_c = pd.to_datetime(m["completion_date_built"])
    assert ((sim_c == built_c) | (sim_c.isna() & built_c.isna())).all()
    assert (pd.to_datetime(m["due_date_sim"]) == pd.to_datetime(m["due_date_built"])).all()
