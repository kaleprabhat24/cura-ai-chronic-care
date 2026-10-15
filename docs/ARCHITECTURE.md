# Cura AI — Architecture and implementation plan

Cura AI is a standalone application for **chronic-disease follow-up risk prioritisation**. It helps
care teams find patients at risk of missing expected follow-up, see why, prioritise outreach and
record what happened. All data are synthetic. The model predicts **follow-up non-completion**, not
deterioration or diagnosis. Results on synthetic data are never presented as clinical evidence.

Detailed specs: [PREDICTION_TARGET.md](PREDICTION_TARGET.md) ·
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) · [DECISIONS.md](DECISIONS.md).

## 0. Guiding decisions
1. ML first: M1–M3 are a pure Python package working on local Parquet files.
2. One feature module (`cura_ml.features`), shared by training and serving.
3. Scores are written by batch scoring runs; every score references a `model_version` and `scored_at`.
4. Three visibly separate classes of information: *simulated prediction*, *recorded clinical data*,
   *recorded outreach outcome*. They are distinct in the database, the API and the UI.

## 1. User journeys (acceptance summary)
| # | Journey | Key acceptance criteria |
|---|---|---|
| J1 | Triage the worklist | Server-side filter/sort/search/pagination; < 500 ms locally; counts consistent with the dashboard |
| J2 | Understand why | Risk card: probability, tier, model version, scored_at, top factors with values, units and direction; "Simulated prediction" label |
| J3 | Review history | Unified, filterable timeline with effective and recorded times; score history |
| J4 | Record outreach | One side-panel form; validation; appears immediately; audited |
| J5 | Outcomes | Recorded outcomes only; immature episodes shown as pending; descriptive, not causal |
| J6 | Model & data quality | Everything read from the stored evaluation report and live queries |
| J7 | FHIR export | Validated R4-oriented Bundle; audited |
| J8 | Access control | 401/403 matrix test across every endpoint and role |

## 2. Stack and layout
See ADR-001. Repository layout:
```
ml/        cura_ml package: synth/, cohort/, features/, modeling/, configs/, tests/, data/, reports/, artifacts/
backend/   (M4+) FastAPI app, Alembic, scripts, tests
frontend/  (M6+) Next.js app
docs/      architecture, decisions, data dictionary, prediction target
```

## 3. Database entities (M4)
`patient`, `condition`, `encounter`, `appointment`, `observation`, `medication_statement`,
`followup_episode`, `model_version`, `scoring_run`, `risk_score` (append-only), `outreach_attempt`,
`app_user`, `audit_event` (append-only). Clinical tables keep both effective and recorded times, as
in the M1 Parquet schema.

## 4. Synthetic data and label
Process-based, event-driven simulator (ADR-003). See PREDICTION_TARGET.md for the episode, eligibility,
censoring, maturity and label rules.

## 5. Features and leakage safeguards (M2)
A declarative feature spec (name, label, unit, source, window, aggregation, missing policy) drives a
single `build_features(records, as_of)`. That function filters on `recorded_at < as_of` before
anything else. Safeguards, each backed by a test:
- the time-travel test (features on full data equal features on data truncated at t0);
- prior-episode outcomes only once matured;
- hidden and outcome tables are unreachable from feature code;
- preprocessing fitted inside the training pipeline;
- a single-feature leak alarm.

## 6. Modelling (M3) — see ADR-014 for the implemented details
**Splits:** a 20% patient holdout, plus a temporal split (train/validation/test by t0 year).
**Baselines:** prevalence and a no-show heuristic. **Models:** Logistic Regression vs XGBoost, with
calibration chosen on the validation year.
**Metrics** (with patient-level bootstrap CIs): PR-AUC (primary), ROC-AUC, Brier, calibration and
precision/recall/lift at outreach capacity, plus subgroup results.
**Selection rule:** XGBoost only if it clearly beats LR without worse calibration.
**Explanations:** LR contributions, or TreeSHAP for XGBoost.
**Output:** a versioned model bundle with model card.

## 7. API (M5+)
REST `/api/v1`: `me`, `dashboard/summary`, `worklist`, `patients/{id}` (+ `timeline`, `risk-scores`,
`fhir`), `episodes/{id}` (+ `outreach-attempts`), `outcomes/summary`, `models`, `data-quality`,
`admin/scoring-runs`, `admin/audit-events`. Each item carries a `provenance` field
(`simulated_prediction` | `recorded_clinical` | `recorded_outcome`). TypeScript types are generated
from OpenAPI.

*Implementation note:* this was the plan. The implemented routes differ in places, e.g.
`/outreach/outcomes`, `/scoring-runs`, `/audit-events` and `/admin/users`, plus the onboarding
routes. The provenance values are `recorded_synthetic_clinical`, `recorded_synthetic_outcome` and
`recorded_outreach_outcome`. The authoritative list is `backend/openapi.json`, summarised in the
README's API reference.

## 8. UI (M6+)
App shell with a permanent "synthetic data" banner. Screens:
- **Worklist** and **Patient 360** (risk card, timeline, outreach panel) — top priority;
- Dashboard;
- Outcomes;
- Model & Data;
- Admin.

## 9. Security, FHIR, audit (M9–M10)
- **Auth:** Supabase Auth JWT verified in FastAPI; roles `coordinator`, `analyst`, `admin`.
- **Supabase tables:** RLS enabled with no policies, so the anon key can't read them directly.
- **Audit:** append-only audit events.
- **FHIR:** Patient/Condition/Encounter/Appointment/Observation/MedicationStatement/RiskAssessment/
  Communication. Structural validation with `fhir.resources`, plus the official HL7 validator in CI.
  Labelled "R4-oriented" until the HL7 validator passes.
- **Compliance:** no HIPAA/GDPR claims.

## 10. Deployment and testing (M11)
**Deployment:** Supabase (Postgres + Auth), Render (API), Vercel (web).
**Tests:** pytest (ml, backend), Vitest, Playwright for J1/J2/J4, GitHub Actions CI.

## Milestones
| M | Scope | Depends on | Status |
|---|---|---|---|
| M1 | Repo, prediction target, synthetic generator, episodes and labels, manifest, report | – | **done** |
| M2 | Feature spec + as_of-safe builder + leakage tests | M1 | **done** |
| M3 | Baselines, LR, XGBoost, calibration, evaluation, explanations, model bundle | M2 | **done** (selected: LR, `cura-fu-1.0.0`) |
| M4 | Database schema (SQLite default, PostgreSQL via DATABASE_URL), loader, batch scoring | M3 | **done** |
| M5 | REST API with dev auth boundary, outreach writes, audit | M4 | **done** (incl. outreach API + data-quality endpoint from M7/M8) |
| M6 | Frontend shell, Worklist, Patient 360 | M5 | **done** |
| M7 | Outreach workflow + Outcomes | M6 | **done** |
| M8 | Model & Data page, data-quality endpoint | M5/M6 | **done** |
| M9 | Supabase Auth, roles, audit viewer | M5 | **done** (code + tests; live project not verified) |
| M10 | FHIR export + validation | M4 | **done** (structural + R4B; HL7 validator not run) |
| M11 | CI, deployment, demo script | all | deployed (Render + Vercel + Supabase; live health and 401 check 2026-10-16); demo script in README §13. **Not done:** GitHub Actions CI, Playwright tests, HL7 validator in CI |
| — | Patient onboarding (ADR-022) | M4–M6 | **done** |
