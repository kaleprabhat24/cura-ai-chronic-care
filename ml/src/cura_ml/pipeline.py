"""End-to-end M1 pipeline: simulate -> persist -> rebuild episodes from persisted records -> manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cura_ml.cohort.episodes import build_episodes
from cura_ml.dataset import load_records, write_records, write_table
from cura_ml.manifest import write_manifest
from cura_ml.synth.generator import generate


def generate_records(cfg: dict[str, Any], out_dir: str | Path) -> None:
    records, hidden = generate(cfg)
    write_records(records, out_dir)
    for table, df in hidden.items():
        write_table(df, out_dir, "hidden", table)


def build_cohort(cfg: dict[str, Any], data_dir: str | Path) -> None:
    # Episodes are built from the *persisted ordinary records only*, exactly as a downstream
    # consumer would see them; hidden tables are never loaded here.
    episodes, labels = build_episodes(load_records(data_dir), cfg)
    write_table(episodes, data_dir, "cohort", "followup_episodes")
    write_table(labels, data_dir, "outcomes", "episode_labels")


def run_all(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    generate_records(cfg, out_dir)
    build_cohort(cfg, out_dir)
    return write_manifest(out_dir, cfg)
