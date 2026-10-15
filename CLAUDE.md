# Cura AI — project notes for Claude

Chronic-disease follow-up risk prioritisation on **synthetic data only**. The model predicts
follow-up non-completion (label = 1 if a review is still incomplete 30 days after its due date), not
deterioration or diagnosis. Never present synthetic results as clinical evidence.

- Plan and milestone status: `docs/ARCHITECTURE.md`. Rules: `docs/PREDICTION_TARGET.md`.
  Schema: `docs/DATA_DICTIONARY.md`. Decisions: `docs/DECISIONS.md` (add an ADR for any
  non-obvious choice).
- Current milestone: **M1 done**; next is M2 (feature spec + as_of-safe feature builder + leakage
  tests). Work one milestone at a time; tests are the definition of done.

## Commands (Windows; project-local venv, since uv was not installed at M1)
```
.venv\Scripts\python -m pytest -q          # all tests
.venv\Scripts\ruff check ml                # lint
.venv\Scripts\cura-ml all                  # regenerate ml/data/v1 + manifest + report (~10 s)
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
