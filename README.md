# Cura AI — chronic-care follow-up prioritisation

> **Synthetic data only. Not for clinical use.** Cura AI predicts whether an expected chronic-care
> follow-up will still be incomplete 30 days after its due date. It does not predict disease
> deterioration and makes no diagnosis.

Status: **Milestone 1 complete**. This covers the synthetic cohort generator, follow-up episodes and
labels, the dataset manifest and the cohort report. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the full plan.

## Quick start (Windows)

Requires Python ≥ 3.12.

**Option A — uv (recommended).** Install uv once (per-user install, no admin needed):
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then, from the project folder:
```
uv sync
uv run pytest
uv run cura-ml all
```

**Option B — standard library venv + pip** (this is how M1 was verified):
```
python -m venv .venv
.venv\Scripts\python -m pip install -e ml pytest ruff
.venv\Scripts\python -m pytest -q
.venv\Scripts\cura-ml all
```

`cura-ml all` writes `ml/data/v1/` (git-ignored Parquet files and `manifest.json`) and
`ml/reports/cohort_v1_summary.md`. Other commands: `generate`, `build-episodes`, `report`
(each takes `--config` and `--out`).

## Repository
```
ml/configs/cohort_v1.yaml    all simulation and episode parameters (seeded)
ml/src/cura_ml/synth/        event-driven synthetic patient simulator
ml/src/cura_ml/cohort/       episode/label construction and cohort report
ml/tests/                    M1 acceptance tests
docs/                        architecture, prediction target, data dictionary, decisions
```
