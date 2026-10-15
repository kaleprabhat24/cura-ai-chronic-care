# Decision log

Short architecture decision records (ADRs). Add new entries at the bottom; never rewrite history.
Supersede an entry with a new one that references it.

**ADR-001 — Technology stack (2026-10-15).** Frontend: Next.js + TypeScript + Tailwind + shadcn/ui,
with TanStack Query/Table and Recharts. Backend: FastAPI + SQLAlchemy 2 + Alembic + Pydantic v2.
ML: scikit-learn, XGBoost, SHAP (offline; XGBoost `pred_contribs` at serving). Database: PostgreSQL
(Docker locally, Supabase in production). Auth: Supabase Auth for identity only, with roles enforced
in FastAPI. Deploy: Vercel + Render. No microservices, queues, vector DBs or LLM scoring.

**ADR-002 — uv workspace; ML independent of infrastructure (2026-10-15).** A uv workspace root holds
member packages (`ml/` now, `backend/` later). M1–M3 use local Parquet files only. *Environment
note:* uv and Python 3.12 were not installed on the development machine at M1, so M1 was verified in
a project-local `.venv` (stdlib venv + pip) on Python 3.13.9. `requires-python = ">=3.12"`.

**ADR-003 — Process-based, event-driven synthetic generator (2026-10-15).** Labels are never written
directly. Hidden traits drive booking, attendance, cancellation and outreach response. Completion
happens only when a simulated appointment is attended. Each patient has an independent random
stream (`SeedSequence(seed).spawn(n)`), so runs are reproducible and adding patients does not change
existing ones.

**ADR-004 — Separate information classes on disk (2026-10-15).** `records/` (known at cutoff),
`cohort/` (known at t0), `outcomes/` (future: targets only) and `hidden/` (latent: validation only).
`load_records()` and the `Records` type can only carry `records/` tables. Episodes are rebuilt from
the *persisted* records, never from simulator state.

**ADR-005 — Completion rules per programme (2026-10-15).** T2D completes on a review encounter,
an attended review appointment or an HbA1c result. HTN completes only on a review encounter or an
attended review appointment. A blood pressure reading alone is too opportunistic to count as a
structured review.

**ADR-006 — 14-day merge window (2026-10-15).** Qualifying events within 14 days of a review start
belong to the same review (e.g. an HbA1c drawn at the visit, or a repeat lab). This prevents one
visit from creating several episodes.

**ADR-007 — Censoring and status precedence (2026-10-15).** A completion inside the window beats
censoring (the outcome is known). An eligible episode without completion and with death/transfer on
or before `window_end_date` is censored. Status order: ineligible → censored → immature → labelled.
Death or transfer on the t0 date makes the episode ineligible.

**ADR-008 — Content hashes in the manifest (2026-10-15).** Each file gets a content hash (of data
as read back from Parquet, independent of encoder bytes) and a file hash. Reproducibility tests
require both to match for the same seed and config.

**ADR-009 — Generator calibration before modelling (2026-10-15).** The booking-lead attendance
penalty is capped (`attendance.long_lead_cap_days`, 120). Without the cap, year-ahead prebooked HTN
reviews pushed label prevalence to ~38% (outside the 15–35% band fixed in the plan). This calibration
used only label prevalence. No model has been trained.

**ADR-010 — Clinic rebooking relative to the lost slot (2026-10-15).** A full-size run showed that
clinic-cancelled appointments were rebooked relative to the *cancellation* date. That occasionally
produced two reviews < 14 days apart, and the simulator log disagreed with the record-based
reconstruction (2 of 27k episodes). Rebooking now targets the original slot + 7–21 days. The shared
test cohort was enlarged to 2,000 patients, and the agreement check also runs on the full dataset.

**ADR-011 — Burn-in period and continuing recall letters (2026-10-15).** Investigating why label
prevalence fell from 38.4% (2020) to 28.7% (2024) showed a change in *which patients generated
episodes*, not a change in behaviour. In 2020, 96% of episodes were patients' first episodes. Every
patient started with a forced baseline review, so disengaged patients were over-represented: mean
hidden engagement rose from 0.02 to 0.43 by 2024. Within each engagement third, prevalence was
roughly flat. A second cause made the mix drift indefinitely: recall letters stopped after 8 attempts,
an absorbing "never contacted again" state. Fix: simulate from 2018-01-01 and treat 2018–2019 as
burn-in. Burn-in records are kept as history; episodes with t0 before
`episodes.first_prediction_date` (2020-01-01) are ineligible (`prediction_before_study_period`).
Recall letters now repeat every 90 days while a review is overdue. Result: prevalence 30.1–30.8% across
2020–2024. Within the low-engagement third it rises slightly (0.474 → 0.489), consistent with the
configured negative attendance drift.

**ADR-012 — HbA1c missingness (2026-10-15).** At a diabetes review the HbA1c may not be drawn
(logit −2.2, more likely with low engagement or instability: missing not at random), or drawn with no
result (2%). No observation is written, so units, timestamps and encounter provenance of the recorded
results are unchanged. Intensification is then skipped. Completion of the review is unaffected (the
encounter still counts). The share of diabetes episodes with an HbA1c in the prior 365 days fell from
100% to 94.7%.

**ADR-013 — Feature design (2026-10-15).** A declarative spec (`features/spec.py`, fs-1.0;
`docs/FEATURES.md`) and one builder. The builder joins each source to episodes and filters on
`recorded_at < t0` before aggregating. Appointment statuses and medicine stops are taken as known at t0.
Prior follow-up outcomes count only when matured before t0, with completion re-derived from events
recorded before t0. Smoothing priors are fixed constants, not fitted. Sex and language are not model
inputs; area deprivation is (it is an observed context variable, and subgroup results are reported).
Missing values: explicit `__missing` indicators, median imputation inside the LR pipeline, and native
NaN handling in XGBoost.

**ADR-014 — Evaluation design and model selection (2026-10-15).** 20% of patients are held out for
the final test (2024). Remaining patients: train 2020–2022, with labels matured before 2023.
Validation 2023 (labels matured before 2024) is split by patient into a tuning half (C, depth, early
stopping) and a calibration half (Platt vs isotonic by grouped CV Brier, model comparison, tier
thresholds). Ranking metrics use raw scores (the worklist order); Brier and calibration use calibrated
probabilities. *Correction:* the first implementation selected XGBoost on a fixed +0.01 validation
PR-AUC margin, which deviated from the approved plan. The plan requires a CI that excludes zero. The
rule now requires the paired patient-level bootstrap 95% CI of XGBoost's validation PR-AUC advantage
to lie above zero. The observed CI was [−0.0007, +0.0284], so logistic regression was selected. The
correction was made after the test metrics had been computed for both models. The test set was not
used by the rule, and LR is also nominally ahead on test (difference CIs include zero).
