"""Idempotent loading of the existing generated synthetic cohort (ml/data/<version>) into the database.

Nothing is simulated here. The loader reads the persisted M1 Parquet files (records/, cohort/,
outcomes/). Hidden simulation traits are never loaded. Each row gets a content hash, and a repeated
run compares hashes:
  inserted       — id not yet in the database
  updated        — id present but content changed in the source files
  skipped        — id present with identical content
  not_in_source  — seeded id no longer in the source files (reported, never deleted automatically)
Outreach entered in the app (`origin = "app"`) is never touched by the loader.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Table, bindparam, select, update
from sqlalchemy.orm import Session, sessionmaker

from cura_api import models as m
from cura_api.db import utcnow
from cura_api.services import audit
from cura_ml.cohort.episodes import as_utc
from cura_ml.dataset import read_table
from cura_ml.manifest import read_manifest
from cura_ml.modeling.train import data_fingerprint

CHUNK = 5000
HISTORICAL = "historical_simulation"
LETTER_NOTE = "Recall letter sent (simulated historical practice letter)."


class SeedError(RuntimeError):
    pass


def _py(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating | float):
        return None if math.isnan(v) else float(v)
    return v


def _rows(df: pd.DataFrame, rename: dict[str, str], ts_cols: list[str]) -> list[dict[str, Any]]:
    df = df.rename(columns=rename).copy()
    for c in ts_cols:
        df[c] = as_utc(df[c])
    cols = list(df.columns)
    out = []
    for values in df.itertuples(index=False, name=None):
        out.append({c: _py(v) for c, v in zip(cols, values, strict=True)})
    return out


def _hash(row: dict[str, Any], columns: list[str]) -> str:
    def norm(v: Any) -> Any:
        if isinstance(v, datetime | date):
            return v.isoformat()
        return v
    payload = json.dumps([norm(row.get(c)) for c in columns], default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _upsert(session: Session, table: Table, key: str, rows: list[dict[str, Any]],
            existing_filter=None) -> dict[str, int]:
    data_cols = [c.name for c in table.columns if c.name != "row_hash"]
    for r in rows:
        r["row_hash"] = _hash(r, data_cols)
    query = select(table.c[key], table.c.row_hash)
    if existing_filter is not None:
        query = query.where(existing_filter)
    existing = dict(session.execute(query).all())
    inserts, updates, skipped = [], [], 0
    for r in rows:
        old = existing.get(r[key])
        if old is None:
            inserts.append(r)
        elif old != r["row_hash"]:
            updates.append(r)
        else:
            skipped += 1
    for i in range(0, len(inserts), CHUNK):
        session.execute(table.insert(), inserts[i:i + CHUNK])
    if updates:
        cols = [c for c in data_cols + ["row_hash"] if c != key]
        stmt = (update(table).where(table.c[key] == bindparam("b_key"))
                .values({c: bindparam(f"b_{c}") for c in cols}))
        params = [{"b_key": r[key], **{f"b_{c}": r.get(c) for c in cols}} for r in updates]
        for i in range(0, len(params), CHUNK):
            session.connection().execute(stmt, params[i:i + CHUNK])
    source_ids = {r[key] for r in rows}
    return {"inserted": len(inserts), "updated": len(updates), "skipped": skipped,
            "not_in_source": len(set(existing) - source_ids), "source_rows": len(rows)}


def _link_outreach_to_episodes(outreach: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """Historical outreach carries patient + programme; attach it to the episode whose review cycle
    it happened in (latest episode of that programme with index event at or before the attempt)."""
    left = outreach[["outreach_id", "patient_id", "programme", "attempted_at"]].copy()
    left["attempted_at"] = as_utc(left["attempted_at"])
    right = episodes[["episode_id", "patient_id", "programme", "index_event_at"]].copy()
    right["index_event_at"] = as_utc(right["index_event_at"])
    left["key"] = left["patient_id"].astype(str) + "|" + left["programme"].astype(str)
    right["key"] = right["patient_id"].astype(str) + "|" + right["programme"].astype(str)
    merged = pd.merge_asof(left.sort_values("attempted_at"), right.sort_values("index_event_at")[
        ["key", "index_event_at", "episode_id"]], left_on="attempted_at", right_on="index_event_at",
        by="key", direction="backward")
    return merged.set_index("outreach_id")["episode_id"].reindex(outreach["outreach_id"])


TableLoader = Callable[[Path], list[dict[str, Any]]]


def _loaders(data_dir: Path) -> list[tuple[str, type[m.Base], str, list[dict[str, Any]], Any]]:
    rec = {t: read_table(data_dir, "records", t) for t in (
        "patients", "conditions", "appointments", "encounters", "observations",
        "medication_statements", "outreach_attempts")}
    episodes = read_table(data_dir, "cohort", "followup_episodes")
    labels = read_table(data_dir, "outcomes", "episode_labels")

    outreach = rec["outreach_attempts"].copy()
    outreach["episode_id"] = _link_outreach_to_episodes(outreach, episodes).to_numpy()
    letter = outreach["outcome"] == "letter_sent"
    outreach["note"] = np.where(letter, LETTER_NOTE, None)
    # The app's outcome vocabulary has no "letter_sent"; a sent letter is recorded as outcome
    # "other" on channel "letter" with an explanatory note (no contact outcome is implied).
    outreach.loc[letter, "outcome"] = "other"
    outreach["origin"] = HISTORICAL
    outreach = outreach.rename(columns={"performed_by": "recorded_by"})
    outreach_cols = ["outreach_id", "patient_id", "episode_id", "programme", "origin", "purpose",
                     "channel", "outcome", "note", "next_action", "next_action_due", "attempted_at",
                     "recorded_at", "recorded_by"]

    return [
        ("patients", m.Patient, "id", _rows(rec["patients"], {"patient_id": "id"},
                                            ["death_recorded_at", "transfer_recorded_at"]), None),
        ("conditions", m.Condition, "id", _rows(rec["conditions"], {"condition_id": "id"},
                                                ["recorded_at"]), None),
        ("appointments", m.Appointment, "id", _rows(
            rec["appointments"], {"appointment_id": "id"},
            ["booked_at", "scheduled_for", "status_recorded_at"]), None),
        ("encounters", m.Encounter, "id", _rows(rec["encounters"], {"encounter_id": "id"},
                                                ["start_at", "end_at", "recorded_at"]), None),
        ("observations", m.Observation, "id", _rows(rec["observations"], {"observation_id": "id"},
                                                    ["effective_at", "recorded_at"]), None),
        ("medication_statements", m.MedicationStatement, "id", _rows(
            rec["medication_statements"], {"medication_statement_id": "id"},
            ["recorded_at", "end_recorded_at"]), None),
        ("followup_episodes", m.FollowupEpisode, "id", _rows(
            episodes, {"episode_id": "id"}, ["index_event_at", "prediction_at"]), None),
        ("episode_outcomes", m.EpisodeOutcome, "episode_id", _rows(labels, {}, ["completed_at"]), None),
        ("outreach_attempts", m.OutreachAttempt, "id", _rows(
            outreach[outreach_cols], {"outreach_id": "id"}, ["attempted_at", "recorded_at"]),
         m.OutreachAttempt.origin == HISTORICAL),
    ]


def seed_database(factory: sessionmaker[Session], data_dir: str | Path,
                  actor: str = "system:cli") -> dict[str, Any]:
    data_dir = Path(data_dir)
    if not (data_dir / "manifest.json").exists():
        raise SeedError(f"no generated dataset at {data_dir}; run `cura-ml all` first")
    manifest = read_manifest(data_dir)
    fingerprint = data_fingerprint(manifest)
    counts: dict[str, dict[str, int]] = {}
    with factory() as session, session.begin():
        for name, model, key, rows, existing_filter in _loaders(data_dir):
            counts[name] = _upsert(session, model.__table__, key, rows, existing_filter)
        load = m.DatasetLoad(dataset_version=manifest["dataset_version"], data_fingerprint=fingerprint,
                             data_cutoff=date.fromisoformat(manifest["data_cutoff"]),
                             dataset_dir=str(data_dir), loaded_at=utcnow(), counts=counts)
        session.add(load)
        session.flush()
        audit.record(session, actor, audit.SYSTEM_ROLE, "dataset.seed", "dataset_load", str(load.id),
                     {"data_fingerprint": fingerprint, "counts": counts})
        load_id = load.id
    return {"dataset_load_id": load_id, "data_fingerprint": fingerprint,
            "dataset_version": manifest["dataset_version"], "data_cutoff": manifest["data_cutoff"],
            "tables": counts}
