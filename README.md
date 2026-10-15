# Cura AI — chronic-care follow-up prioritisation

> **Synthetic data only. Not for clinical use.** Cura AI predicts whether an expected chronic-care
> follow-up will still be incomplete 30 days after its due date. It does not predict disease
> deterioration and makes no diagnosis.

Status: **Milestones 1–3 complete**. This covers:
- the synthetic cohort, with follow-up episodes and labels;
- leakage-safe longitudinal features;
- baseline, logistic-regression and XGBoost models with calibration, evaluation and per-prediction
  explanations;
- an inference-ready model bundle.

All results are a synthetic-data evaluation, not clinical validation; see the
[model report](ml/reports/model_cura-fu-1.0.0.md). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
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

Then build features, train, and use the saved model:
```
.venv\Scripts\cura-ml features
.venv\Scripts\cura-ml train
.venv\Scripts\cura-ml evaluate
.venv\Scripts\cura-ml predict --limit 1
```

`cura-ml all` writes `ml/data/v1/` (git-ignored Parquet files and `manifest.json`) and
`ml/reports/cohort_v1_summary.md`. `cura-ml train` writes the model bundle to
`ml/artifacts/models/cura-fu-1.0.0/` and the report to `ml/reports/model_cura-fu-1.0.0.md`.
`predict --episode-id <id>` scores a specific eligible episode.

## Repository
```
ml/configs/cohort_v1.yaml    simulation and episode parameters (seeded)
ml/configs/train_v1.yaml     splits, tuning grids, calibration, selection rule, capacity, tiers
ml/src/cura_ml/synth/        event-driven synthetic patient simulator
ml/src/cura_ml/cohort/       episode/label construction and cohort report
ml/src/cura_ml/features/     declarative feature spec + leakage-safe feature builder
ml/src/cura_ml/modeling/     splits, models, calibration, metrics, explanations, bundle, reports
ml/artifacts/models/         saved, versioned model bundles
ml/tests/                    acceptance, leakage and modelling tests
docs/                        architecture, prediction target, features, data dictionary, decisions
```
