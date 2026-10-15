"""Dataset layout, Parquet I/O and content hashing.

The on-disk layout enforces the separation between information classes:

    records/   ordinary synthetic records as known at the data cutoff (feature inputs)
    cohort/    follow-up episode definitions (known at prediction time)
    outcomes/  episode outcomes and target labels (future relative to each prediction)
    hidden/    latent simulation traits and simulator internals (never ML inputs)

`load_records` is the only loader intended for feature engineering and it reads `records/` only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

RECORD_TABLES = (
    "patients", "conditions", "appointments", "encounters", "observations",
    "medication_statements", "outreach_attempts",
)
HIDDEN_TABLES = ("simulation_traits", "simulator_episode_log")
COHORT_TABLES = ("followup_episodes",)
OUTCOME_TABLES = ("episode_labels",)

LAYOUT = {
    "records": RECORD_TABLES,
    "hidden": HIDDEN_TABLES,
    "cohort": COHORT_TABLES,
    "outcomes": OUTCOME_TABLES,
}


@dataclass(frozen=True)
class Records:
    """Ordinary records only. Deliberately has no field for hidden traits or labels."""

    patients: pd.DataFrame
    conditions: pd.DataFrame
    appointments: pd.DataFrame
    encounters: pd.DataFrame
    observations: pd.DataFrame
    medication_statements: pd.DataFrame
    outreach_attempts: pd.DataFrame


assert tuple(f.name for f in fields(Records)) == RECORD_TABLES


def table_path(data_dir: str | Path, group: str, table: str) -> Path:
    if table not in LAYOUT[group]:
        raise KeyError(f"{table!r} is not a {group} table")
    return Path(data_dir) / group / f"{table}.parquet"


def write_table(df: pd.DataFrame, data_dir: str | Path, group: str, table: str) -> Path:
    path = table_path(data_dir, group, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_table(data_dir: str | Path, group: str, table: str) -> pd.DataFrame:
    return pd.read_parquet(table_path(data_dir, group, table))


def load_records(data_dir: str | Path) -> Records:
    return Records(**{t: read_table(data_dir, "records", t) for t in RECORD_TABLES})


def write_records(records: Records, data_dir: str | Path) -> None:
    for t in RECORD_TABLES:
        write_table(getattr(records, t), data_dir, "records", t)


def content_hash(df: pd.DataFrame) -> str:
    """Hash of table *content* (columns, dtypes, values, row order), independent of Parquet bytes."""
    h = hashlib.sha256()
    h.update("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode("utf-8"))
    if len(df):
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
