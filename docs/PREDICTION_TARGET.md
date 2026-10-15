# Prediction target (v1)

Implemented in `ml/src/cura_ml/cohort/episodes.py`; parameters in `ml/configs/cohort_v1.yaml`
(`programmes`, `episodes`). Verified by `ml/tests/test_labels.py`.

**The model predicts follow-up non-completion — not disease deterioration, not a diagnosis.**
All data are synthetic.

## 1. Data snapshot

The dataset is a snapshot of everything **recorded by the end of the data cutoff day**
(`simulation.data_cutoff`, 2024-12-31 in v1). A record is known only if its `recorded_at`
(or `status_recorded_at`) is on or before the cutoff. Future appointments appear only as bookings
(or cancellations) recorded before the cutoff.

## 2. Programmes

| Programme | Condition (SNOMED CT) | Review interval | Completion events |
|---|---|---|---|
| `t2d` | 44054006 Diabetes mellitus type 2 | 182 days | finished `review_t2d` encounter, attended `review_t2d` appointment, or HbA1c observation (LOINC 4548-4) |
| `htn` | 38341003 Hypertensive disorder | 365 days | finished `review_htn` encounter or attended `review_htn` appointment (BP alone does **not** count — it is measured opportunistically at many visits) |

## 3. Follow-up episodes

1. Collect the qualifying completion events for each patient × programme that were recorded by the cutoff.
2. Order them by effective time (ties: encounter, then appointment, then observation).
3. **Merge window:** an event starts a new review only if it is ≥ `merge_window_days` (14) after the
   previous review start; otherwise it belongs to the same review (e.g. the HbA1c drawn at the visit).
4. Each review start is the **index event** of one episode and the **completion event** of the previous one.

For an episode with index event at time *I*:

| Quantity | Definition (v1 values) |
|---|---|
| `due_date` | date(*I*) + interval |
| `prediction_at` (t0) | `due_date` − 14 days, 00:00 UTC |
| `window_end_date` | `due_date` + 30 days (inclusive) |
| Observation window for features (M2) | records with `recorded_at` < t0, 730 days back |

## 4. Eligibility at t0 (first failing rule is reported)

| # | `ineligible_reason` | Rule |
|---|---|---|
| 0 | `prediction_before_study_period` | t0 before `episodes.first_prediction_date` (2020-01-01). The simulated history starts on 2018-01-01; 2018–2019 is burn-in, used only as history (ADR-011) |
| 1 | `prediction_after_cutoff` | t0 is after the data cutoff day |
| 2 | `deceased_before_prediction` | known death date ≤ t0 date |
| 3 | `transferred_before_prediction` | known transfer-out date ≤ t0 date |
| 4 | `condition_not_recorded` | programme condition not recorded before t0 |
| 5 | `no_prior_encounter` | no encounter recorded before t0 |
| 6 | `completed_before_prediction` | completion event effective before t0 |

Eligible episodes with < 180 days between the patient's first known encounter and t0 are flagged
`is_cold_start` (they stay eligible). `history_days` = whole days from that first encounter to t0.

## 5. Outcomes and labels (`outcomes/episode_labels.parquet`)

- `completed_in_window` — completion effective in [t0, `window_end_date`] (dates inclusive).
- **Censoring:** an eligible episode with no completion in the window whose patient died or
  transferred out on or before `window_end_date` is `censored`. A completion inside the window takes
  precedence: the outcome is then known (label 0) and the episode is not censored.
- **Maturity:** `mature` when cutoff ≥ `window_end_date` + `recording_lag_days` (7).
- `label_status`, evaluated in order: `ineligible` → `censored` → `immature` → `labelled`.
- **`label` = 1** if follow-up was still incomplete at `window_end_date`, **0** if completed in the
  window. Present **only** for `labelled` episodes; null otherwise.

## 6. Leakage notes for M2 (feature engineering)

- `followup_episodes` lists every episode, including later ones. A later episode's `index_event_at`
  is the earlier episode's completion. Features for an episode at t0 may only use **other** episodes
  whose index event was recorded before t0, and prior-episode outcomes only once matured
  (`window_end_date + lag < t0`).
- `outcomes/` tables are targets only and never feature inputs. `hidden/` is never read by
  feature code.
- Outreach after t0 influences outcomes. The label therefore means "incomplete **under the
  historical care policy**". It is not a causal estimate of what outreach achieves.
