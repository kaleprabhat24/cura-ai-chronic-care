"""Dataset manifest: seed, config hash, per-file content hashes and row counts."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from cura_ml import GENERATOR_VERSION
from cura_ml.config import config_hash
from cura_ml.dataset import LAYOUT, content_hash, file_hash, table_path

MANIFEST_NAME = "manifest.json"


def build_manifest(data_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for group, tables in LAYOUT.items():
        for table in tables:
            path = table_path(data_dir, group, table)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            files[f"{group}/{table}.parquet"] = {
                "rows": len(df),
                "columns": list(df.columns),
                # content hash is computed from the data as read back from disk, so it is
                # independent of Parquet encoder details; file hash detects byte-level changes.
                "content_sha256": content_hash(df),
                "file_sha256": file_hash(path),
            }
    return {
        "dataset_version": cfg["dataset_version"],
        "generator_version": GENERATOR_VERSION,
        "seed": cfg["seed"],
        "config_sha256": config_hash(cfg),
        "start_date": cfg["simulation"]["start_date"],
        "data_cutoff": cfg["simulation"]["data_cutoff"],
        "n_patients": cfg["simulation"]["n_patients"],
        "synthetic_data_only": True,
        "files": files,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "config": cfg,
    }


def write_manifest(data_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = build_manifest(data_dir, cfg)
    path = Path(data_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return manifest


def data_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash identifying the ordinary dataset (config + every non-hidden file's content)."""
    parts = {"config_sha256": manifest["config_sha256"],
             **{k: v["content_sha256"] for k, v in sorted(manifest["files"].items())
                if not k.startswith("hidden/")}}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def read_manifest(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
