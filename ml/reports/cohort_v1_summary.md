# Cura AI synthetic cohort summary — v1

> Generated programmatically by `cura-ml report` from the persisted dataset. **All data are synthetic.** Nothing here is evidence of real-world clinical behaviour or model performance.

## Provenance

| item | value |
|---|---|
| dataset_version | v1 |
| generator_version | 0.1.0 |
| seed | 20251015 |
| config_sha256 | eed3baf932530a27… |
| history start | 2018-01-01 |
| data cutoff (simulated today) | 2024-12-31 |
| python / numpy / pandas / pyarrow | 3.13.9 / 2.5.4 / 3.0.6 / 26.0.0 |

## Record tables

| table | rows |
|---|---|
| records/patients | 5,000 |
| records/conditions | 6,383 |
| records/appointments | 50,389 |
| records/encounters | 62,708 |
| records/observations | 187,162 |
| records/medication_statements | 11,531 |
| records/outreach_attempts | 50,125 |
| cohort/followup_episodes.parquet | 36,197 |
| hidden/simulation_traits.parquet | 5,000 |
| hidden/simulator_episode_log.parquet | 36,197 |
| outcomes/episode_labels.parquet | 36,197 |

## Patients

| measure | value |
|---|---|
| patients | 5,000 |
| age at cutoff, median (IQR) | 68.2 (55.3–82.0) |
| female share | 48.8% |
| registered after history start | 15.7% |
| deaths recorded by cutoff | 267 |
| transfers out recorded by cutoff | 736 |
| with type 2 diabetes recorded | 2,978 |
| with hypertension recorded | 3,405 |
| with both conditions | 1,431 |
| with no programme condition recorded | 48 |

| deprivation_quintile | count | share |
|---|---|---|
| 3 | 1,069 | 0.214 |
| 1 | 1,059 | 0.212 |
| 5 | 986 | 0.197 |
| 4 | 953 | 0.191 |
| 2 | 933 | 0.187 |

## Care-process records

### Appointments by status

| status | count | share |
|---|---|---|
| attended | 36,197 | 0.718 |
| no_show | 8,909 | 0.177 |
| cancelled_clinic | 1,978 | 0.039 |
| cancelled_patient | 1,815 | 0.036 |
| booked | 1,490 | 0.030 |

### Appointments by booking source

| booking_source | count | share |
|---|---|---|
| patient | 25,081 | 0.498 |
| practice_prebook | 14,386 | 0.285 |
| practice | 6,317 | 0.125 |
| outreach | 2,965 | 0.059 |
| clinic_rebook | 1,640 | 0.033 |

### Encounters by type

| encounter_type | count | share |
|---|---|---|
| acute | 26,511 | 0.423 |
| review_t2d | 21,414 | 0.341 |
| review_htn | 14,783 | 0.236 |

### Observations by code

| code | display | unit | count | median_value |
|---|---|---|---|---|
| 29463-7 | Body weight | kg | 44,114 | 86.1 |
| 4548-4 | Hemoglobin A1c/Hemoglobin.total in Blood | % | 18,907 | 7.3 |
| 8462-4 | Diastolic blood pressure | mm[Hg] | 48,815 | 84.0 |
| 8480-6 | Systolic blood pressure | mm[Hg] | 48,815 | 143.0 |
| 8867-4 | Heart rate | /min | 26,511 | 76.0 |

### Medication statements

| display | statements | ended_by_cutoff |
|---|---|---|
| amlodipine | 1,132 | 43 |
| atorvastatin | 1,768 | 213 |
| empagliflozin | 449 | 29 |
| glipizide | 1,074 | 90 |
| hydrochlorothiazide | 378 | 11 |
| insulin glargine | 159 | 11 |
| lisinopril | 3,430 | 210 |
| metformin | 3,141 | 369 |

### Historical outreach attempts

| purpose | outcome | count |
|---|---|---|
| post_due_chase | left_message | 1,742 |
| post_due_chase | no_answer | 2,699 |
| post_due_chase | reached_booked | 2,764 |
| post_due_chase | reached_declined | 1,567 |
| post_due_chase | wrong_number | 532 |
| pre_due_reminder | left_message | 114 |
| pre_due_reminder | no_answer | 229 |
| pre_due_reminder | reached_booked | 201 |
| pre_due_reminder | reached_declined | 139 |
| pre_due_reminder | wrong_number | 32 |
| recall_letter | letter_sent | 40,106 |

## Follow-up episodes

Episodes are reconstructed from ordinary records only. Prediction timestamp t0 = due date − 14 days; label window [t0, due + 30 days]; outcome mature when cutoff ≥ due + 37 days.

| measure | value |
|---|---|
| episodes (all) | 36,197 |
| eligible at t0 | 23,514 |
| eligible, cold start (< 180 days history) | 173 |
| eligible, labelled (mature, uncensored) | 22,858 |
| eligible, censored | 48 |
| eligible, immature at cutoff | 608 |
| eligible, no completion recorded by cutoff (any year) | 2,268 |
| currently open at cutoff (eligible, immature, not yet completed) — worklist candidates | 332 |

### Ineligibility reasons

| reason | count | share |
|---|---|---|
| prediction_before_study_period | 8,377 | 0.660 |
| prediction_after_cutoff | 3,008 | 0.237 |
| completed_before_prediction | 761 | 0.060 |
| transferred_before_prediction | 398 | 0.031 |
| deceased_before_prediction | 139 | 0.011 |

### Label status by programme

| programme | censored | immature | ineligible | labelled |
|---|---|---|---|---|
| htn | 20 | 254 | 5,212 | 9,297 |
| t2d | 28 | 354 | 7,471 | 13,561 |

## Label prevalence (labelled episodes)

Overall prevalence of *follow-up incomplete at due + 30 days*: **30.5%** (n = 22,858). Configured plausibility band 15.0%–35.0%: **PASS**.

| programme | labelled | prevalence |
|---|---|---|
| htn | 9,297 | 0.325 |
| t2d | 13,561 | 0.291 |

| prediction_year | labelled | prevalence |
|---|---|---|
| 2020 | 4,917 | 0.301 |
| 2021 | 4,911 | 0.305 |
| 2022 | 4,708 | 0.306 |
| 2023 | 4,519 | 0.308 |
| 2024 | 3,803 | 0.306 |

## Outcome maturity (eligible episodes by prediction year)

| prediction_year | censored | immature | labelled |
|---|---|---|---|
| 2020 | 6 | 0 | 4,917 |
| 2021 | 10 | 0 | 4,911 |
| 2022 | 15 | 0 | 4,708 |
| 2023 | 9 | 0 | 4,519 |
| 2024 | 8 | 608 | 3,803 |

## Missingness

### Null share in record columns (columns with any nulls)

| table | column | null_share |
|---|---|---|
| patients | death_date | 0.947 |
| patients | death_recorded_at | 0.947 |
| patients | transfer_out_date | 0.853 |
| patients | transfer_recorded_at | 0.853 |
| encounters | appointment_id | 0.423 |
| medication_statements | end_date | 0.915 |
| medication_statements | end_recorded_at | 0.915 |
| outreach_attempts | next_action_due | 0.059 |

Most nulls are structural (e.g. `death_date` for living patients, `appointment_id` for walk-in acute encounters, `end_date` for ongoing medicines).

### Information available at prediction time (eligible episodes)

Computed only from records with `recorded_at` (or `status_recorded_at` / `booked_at`) strictly before t0.

| measure | share of eligible episodes |
|---|---|
| T2D episodes: HbA1c recorded in prior 365 days | 94.7% |
| All episodes: systolic BP recorded in prior 365 days | 97.0% |
| All episodes: no-show recorded in prior 365 days | 15.9% |
| All episodes: outreach attempt recorded in prior 365 days | 37.1% |
| All episodes: review appointment already booked for [t0, due] at t0 | 44.0% |

## Separation of information classes

- `records/` — ordinary records as known at the cutoff (future model inputs).
- `cohort/` — episode definitions and eligibility, known at t0.
- `outcomes/` — completion, censoring, maturity and labels (targets only, never features).
- `hidden/` — latent simulation traits and the simulator's episode log (validation only).
