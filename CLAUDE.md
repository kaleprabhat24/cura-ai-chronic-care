# Cura AI — project notes for Claude

Chronic-disease follow-up risk prioritisation on **synthetic data only**. The model predicts
follow-up non-completion (label = 1 if a review is still incomplete 30 days after its due date), not
deterioration or diagnosis. Never present synthetic results as clinical evidence.

- Plan and milestone status: `docs/ARCHITECTURE.md`. Rules: `docs/PREDICTION_TARGET.md`.
  Schema: `docs/DATA_DICTIONARY.md`. Decisions: `docs/DECISIONS.md` (add an ADR for any
  non-obvious choice).
- Status: **M1–M3 done** (data, features, models). Selected model: logistic regression, bundle
  `ml/artifacts/models/cura-fu-1.0.0/`, report `ml/reports/model_cura-fu-1.0.0.md`. Next: M4
  (Postgres + batch scoring). Work one milestone at a time; tests are the definition of done.
- Features: `docs/FEATURES.md` is generated from `ml/src/cura_ml/features/spec.py`. Never edit it by hand.

## Commands (Windows; project-local venv, since uv was not installed)
```
.venv\Scripts\python -m pytest -q          # all tests (~1 min)
.venv\Scripts\ruff check ml                # lint
.venv\Scripts\cura-ml all                  # regenerate ml/data/v1 + manifest + cohort report (~12 s)
.venv\Scripts\cura-ml features             # feature table + docs/FEATURES.md (~7 s)
.venv\Scripts\cura-ml train                # train/evaluate/select; write bundle + model report (~20 s)
.venv\Scripts\cura-ml evaluate             # reload bundle and recompute test metrics (must match)
.venv\Scripts\cura-ml predict --limit 1    # score + explain episodes open at the cutoff
```
With uv installed: `uv sync` then `uv run pytest`, `uv run cura-ml all`.

## Invariants — do not break
- Data layout: `records/` (known at cutoff), `cohort/` (known at t0), `outcomes/` (targets only),
  `hidden/` (latent: tests only). Feature code reads `records/` (+ episode definitions) only.
- Every feature filters on `recorded_at < t0` (or `status_recorded_at` / `booked_at`).
  Later episodes in `followup_episodes` leak completion dates. Use them only if recorded before t0.
- All simulation parameters live in `ml/configs/cohort_v1.yaml`. Changing one changes the config
  hash, so record why in DECISIONS.md.
- Generator determinism: one `SeedSequence` child per patient. Keep random draws per patient in a
  fixed order.
- Times are UTC; observations carry UCUM units validated against `synth/catalog.py`.

## Conventions
Python ≥ 3.12, pandas, line length 110, ruff (`E,F,I,B,UP,SIM`). Match the existing module style:
module docstring explaining intent, small functions, comments only where the "why" isn't obvious.
