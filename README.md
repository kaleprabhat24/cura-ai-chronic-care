# Cura AI — chronic-care follow-up prioritisation

> **Synthetic demonstration data only. Not for clinical use.** Cura AI estimates whether an expected
> chronic-care follow-up (type 2 diabetes or hypertension review) will still be incomplete 30 days
> after its due date. It does not predict disease deterioration, diagnose, or recommend treatment.
> Model results are a synthetic-data evaluation, not clinical validation.

Cura AI has four parts:
- **`ml/`**: synthetic cohort generator, leakage-safe feature pipeline, models and a saved bundle.
- **`backend/`**: FastAPI + SQLAlchemy + Alembic API with batch scoring, outreach, audit,
  Supabase-ready authentication and FHIR export.
- **`frontend/`**: Next.js + TypeScript + Tailwind + shadcn/ui-style components + Recharts.
- **`docs/`**: architecture, decisions, prediction target, features, data dictionary, FHIR, Supabase
  and deployment.

## Application
| Page | What it does | Roles |
|---|---|---|
| Dashboard | Open follow-ups by status, predictive priority distribution, due-date load, outreach and completion figures, model/data status, recent activity | all |
| Worklist | Server-side paginated/sorted/filtered queue. Tabs separate **predictive** risk, **overdue** follow-ups (recorded outcomes; historical scores) and follow-ups with **no usable score**. Each score shows prediction time, scoring time and model version | coordinator, admin |
| Patient 360 | Summary and conditions, episode score context, the model's actual contribution explanation, score history across runs, HbA1c/BP trends, unified timeline, outreach history and form, FHIR export | coordinator, admin |
| Outreach | Pending next actions, recorded attempts (all 7 outcome categories), follow-up from a side sheet | coordinator, admin |
| Outcomes | Monthly attempts, contact outcomes, completion by year, condition and outreach exposure (labelled descriptive, not causal) | all |
| Model & data | Model version and algorithm, test metrics with bootstrap intervals for all four models, calibration, feature importance, tier results, model card and report. Analyst/admin also see data-quality checks, missingness and scoring runs | all (details: analyst, admin) |
| Admin & export | FHIR export of a selected patient, plus the audit log and user roles for admins | coordinator (export), admin |

The API enforces every permission. The frontend only hides navigation.

## Local setup (Windows, PowerShell)
Prerequisites: Python ≥ 3.12 (verified on 3.13.9) and Node.js 20+ (verified on 22.16).

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ml -e "backend[postgres]" pytest ruff httpx
cd frontend
npm install
cd ..
```
(With [uv](https://docs.astral.sh/uv/) installed instead: `uv sync`.)

### 1. Synthetic data (regenerate any time; deterministic for a given seed)
```
.venv\Scripts\cura-ml all
```
This writes `ml/data/v1/` (git-ignored) and `ml/reports/cohort_v1_summary.md`.

### 2. Model: load the saved bundle or retrain
The committed bundle `ml/artifacts/models/cura-fu-1.0.0/` is used as-is. To retrain and re-evaluate
(about 20 s), run:
```
.venv\Scripts\cura-ml features
.venv\Scripts\cura-ml train
.venv\Scripts\cura-ml evaluate
```

### 3. Database, cohort and scoring
SQLite at `backend/var/cura_dev.db` by default. No server or Docker is needed.
```
.venv\Scripts\cura-api db upgrade
.venv\Scripts\cura-api seed
.venv\Scripts\cura-api score
```
`seed` is idempotent and reports inserted/updated/skipped rows. `score` adds a new immutable scoring
snapshot and keeps earlier ones.

### 4. Start the API and the frontend (two terminals)
```
.venv\Scripts\cura-api serve
```
```
npm --prefix frontend run dev
```
Open http://localhost:3000 and sign in with the **development identity picker** (choose a role). API
docs are at http://127.0.0.1:8000/docs. For a production-mode frontend build use
`npm --prefix frontend run build` then `npm --prefix frontend run start`.

Configuration is optional locally. See [.env.example](.env.example) (backend) and
[frontend/.env.example](frontend/.env.example). Both list variable names only.

## Supabase (PostgreSQL + Auth)
Local development needs no Supabase. To use it:
1. Follow [docs/SUPABASE_SETUP.md](docs/SUPABASE_SETUP.md) to create the project and run the migrations.
2. Set `AUTH_MODE=supabase` and `SUPABASE_URL` on the backend.
3. Set `NEXT_PUBLIC_AUTH_MODE=supabase`, `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   on the frontend.
4. Provision users with `cura-api users add EMAIL --role ROLE`.

The API verifies Supabase access tokens (signature via the project's JWKS, issuer, audience, expiry,
claims) and takes the role from `app_users`. Missing or invalid credentials fail closed.

## Tests and checks
```
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff check ml backend
npm --prefix frontend run lint
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```
- **Python tests:** synthetic data, leakage, models, persistence, API, score status, Supabase token
  validation with locally generated keys, FHIR structure, PostgreSQL migration rendering.
- **Frontend tests:** session handling, API client error mapping, score presentation.

After changing backend schemas, regenerate the API contract and the frontend types:
```
.venv\Scripts\cura-api openapi
npm --prefix frontend run gen:api
```

## Deployment
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md):
- **Backend:** Render, via [render.yaml](render.yaml).
- **Frontend:** Vercel, with root directory `frontend`.
- **Database and Auth:** Supabase.

Required variables:

| Where | Variables |
|---|---|
| Render (API) | `ENVIRONMENT=production`, `AUTH_MODE=supabase`, `SUPABASE_URL`, `DATABASE_URL`, `DATABASE_SSLMODE=require`, `CORS_ORIGINS` (optional: `SUPABASE_JWT_SECRET` for legacy HS256 projects, `DATABASE_DISABLE_PREPARED_STATEMENTS` for the transaction pooler) |
| Vercel (web) | `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_AUTH_MODE=supabase`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_APP_ENV=production` |

## Known limitations
- **Data and evaluation.** All data is synthetic. Model metrics describe recovery of simulated
  structure, not clinical performance. Outreach statistics are descriptive.
- **"Today" is the data cutoff.** The dataset's simulated "today" is 2024-12-31. Many open
  follow-ups are long overdue: they are shown as recorded outcomes needing operational follow-up,
  and their stored scores are labelled historical.
- **Supabase not verified live.** Supabase Auth/PostgreSQL integration is implemented and tested
  with local keys and offline SQL rendering. It has not been run against a live project.
- **FHIR.** Export is checked structurally and with `fhir.resources` R4B models. It has not been run
  through the official HL7 validator, so formal R4 conformance is not claimed (see
  [docs/FHIR_EXPORT.md](docs/FHIR_EXPORT.md)).
- **Development sign-in.** It is for local use only and is refused in production.
- **Not deployed.** Nothing has been deployed by the author.
- **No compliance claims.** No regulatory compliance (HIPAA, GDPR, etc.) is claimed.

## Repository
```
ml/                 synthetic data, features, models (cura_ml) + configs, reports, saved bundles, tests
backend/            FastAPI app (cura_api): models, migrations, services, API routers, tests, openapi.json
frontend/           Next.js app: src/app (pages), src/components, src/lib (API client, auth, formatting)
docs/               ARCHITECTURE, DECISIONS, PREDICTION_TARGET, FEATURES, DATA_DICTIONARY, FHIR_EXPORT,
                    SUPABASE_SETUP, DEPLOYMENT
render.yaml         Render blueprint for the API
```
