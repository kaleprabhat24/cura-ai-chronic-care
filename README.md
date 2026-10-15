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

## Backend API (FastAPI)

The backend runs locally with **no database server, Docker or cloud account**. By default it uses a
SQLite file at `backend/var/cura_dev.db` (git-ignored). For PostgreSQL, set `DATABASE_URL`
(see [.env.example](.env.example)) and install the driver with `pip install -e "backend[postgres]"`.
Prerequisite: the dataset and model bundle from the steps above (`cura-ml all`, then `cura-ml train`
if `ml/artifacts/models/cura-fu-1.0.0/` is missing).

Install it once (into the same `.venv`):
```
.venv\Scripts\python -m pip install -e backend httpx
```

Optionally copy the configuration template (every value has a working default):
```
copy .env.example .env
```

1. Initialise the database (applies the Alembic migrations):
   ```
   .venv\Scripts\cura-api db upgrade
   ```
2. Load the generated synthetic cohort. Safe to repeat: it reports inserted, updated and skipped
   rows, and never touches outreach entered in the app:
   ```
   .venv\Scripts\cura-api seed
   ```
3. Batch-score eligible episodes with the saved model. Each run is kept as a new snapshot; use
   `--scope open` to score only episodes still open at the data cutoff:
   ```
   .venv\Scripts\cura-api score
   ```
4. Start the API, then open http://127.0.0.1:8000/docs for the interactive OpenAPI docs:
   ```
   .venv\Scripts\cura-api serve
   ```
5. Run all tests (ML and backend):
   ```
   .venv\Scripts\python -m pytest -q
   ```

To regenerate the committed OpenAPI document, run `.venv\Scripts\cura-api openapi`. It writes
[backend/openapi.json](backend/openapi.json).

**Calling the API locally:** only a *development* authentication mode exists. Send the
`X-Dev-User: <id>` and `X-Dev-Role: coordinator | analyst | admin` headers. The mode refuses to run
when `ENVIRONMENT=production`. Production (Supabase) authentication is **not implemented yet**.
```
curl -H "X-Dev-User: coord1" -H "X-Dev-Role: coordinator" "http://127.0.0.1:8000/api/v1/worklist?tier=high&page_size=5"
```

| Endpoint (prefix `/api/v1`) | Roles |
|---|---|
| `GET /health` | public |
| `GET /me`, `GET /dashboard/summary`, `GET /outreach/outcomes`, `GET /models/current` | all |
| `GET /worklist`, `GET /patients/{id}`, `GET /patients/{id}/timeline`, `GET /patients/{id}/outreach-attempts`, `GET /episodes/{id}`, `GET /episodes/{id}/risk-scores`, `GET /risk-scores/{id}`, `GET`/`POST /episodes/{id}/outreach-attempts` | coordinator, admin |
| `GET /models/{version}`, `GET /scoring-runs`, `GET /scoring-runs/{id}`, `GET /data-quality` | analyst, admin |
| `GET /audit-events` | admin |

## Repository
```
ml/configs/cohort_v1.yaml    simulation and episode parameters (seeded)
ml/configs/train_v1.yaml     splits, tuning grids, calibration, selection rule, capacity, tiers
ml/src/cura_ml/synth/        event-driven synthetic patient simulator
ml/src/cura_ml/cohort/       episode/label construction and cohort report
ml/src/cura_ml/features/     declarative feature spec + leakage-safe feature builder
ml/src/cura_ml/modeling/     splits, models, calibration, metrics, explanations, bundle, reports
ml/artifacts/models/         saved, versioned model bundles
backend/src/cura_api/        FastAPI app, ORM models, Alembic migrations, seeding, batch scoring
backend/tests/               backend persistence, scoring and API tests
backend/openapi.json         generated OpenAPI document
ml/tests/                    acceptance, leakage and modelling tests
docs/                        architecture, prediction target, features, data dictionary, decisions
```
