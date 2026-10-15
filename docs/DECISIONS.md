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
