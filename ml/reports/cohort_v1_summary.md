# Cura AI synthetic cohort summary — v1

> Generated programmatically by `cura-ml report` from the persisted dataset. **All data are synthetic.** Nothing here is evidence of real-world clinical behaviour or model performance.

## Provenance

| item | value |
|---|---|
| dataset_version | v1 |
| generator_version | 0.1.0 |
| seed | 20251015 |
| config_sha256 | 7dd25a83ef9f2224… |
| history start | 2020-01-01 |
| data cutoff (simulated today) | 2024-12-31 |
| python / numpy / pandas / pyarrow | 3.13.9 / 2.5.4 / 3.0.6 / 26.0.0 |

## Record tables

| table | rows |
|---|---|
| records/patients | 5,000 |
| records/conditions | 6,421 |
| records/appointments | 36,977 |
| records/encounters | 46,463 |
| records/observations | 140,729 |
| records/medication_statements | 10,634 |
| records/outreach_attempts | 28,926 |
| cohort/followup_episodes.parquet | 27,078 |
| hidden/simulation_traits.parquet | 5,000 |
| hidden/simulator_episode_log.parquet | 27,078 |
| outcomes/episode_labels.parquet | 27,078 |

## Patients

| measure | value |
|---|---|
| patients | 5,000 |
| age at cutoff, median (IQR) | 66.2 (53.3–80.0) |
| female share | 48.8% |
| registered after history start | 15.7% |
| deaths recorded by cutoff | 180 |
| transfers out recorded by cutoff | 532 |
| with type 2 diabetes recorded | 2,974 |
| with hypertension recorded | 3,447 |
| with both conditions | 1,447 |
| with no programme condition recorded | 26 |

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
| attended | 27,078 | 0.732 |
| no_show | 5,797 | 0.157 |
| booked | 1,525 | 0.041 |
| cancelled_clinic | 1,355 | 0.037 |
| cancelled_patient | 1,222 | 0.033 |

### Appointments by booking source

| booking_source | count | share |
|---|---|---|
| patient | 16,611 | 0.449 |
| practice_prebook | 10,762 | 0.291 |
| practice | 6,364 | 0.172 |
| outreach | 2,105 | 0.057 |
| clinic_rebook | 1,135 | 0.031 |

### Encounters by type

| encounter_type | count | share |
|---|---|---|
| acute | 19,385 | 0.417 |
| review_t2d | 15,958 | 0.343 |
| review_htn | 11,120 | 0.239 |

### Observations by code

| code | display | unit | count | median_value |
|---|---|---|---|---|
| 29463-7 | Body weight | kg | 32,898 | 86.3 |
| 4548-4 | Hemoglobin A1c/Hemoglobin.total in Blood | % | 15,936 | 7.3 |
| 8462-4 | Diastolic blood pressure | mm[Hg] | 36,255 | 83.0 |
| 8480-6 | Systolic blood pressure | mm[Hg] | 36,255 | 143.0 |
| 8867-4 | Heart rate | /min | 19,385 | 76.0 |

### Medication statements

| display | statements | ended_by_cutoff |
|---|---|---|
| amlodipine | 853 | 24 |
| atorvastatin | 1,779 | 131 |
| empagliflozin | 332 | 13 |
| glipizide | 863 | 53 |
| hydrochlorothiazide | 214 | 3 |
| insulin glargine | 87 | 3 |
| lisinopril | 3,435 | 127 |
| metformin | 3,071 | 251 |

### Historical outreach attempts

| purpose | outcome | count |
|---|---|---|
| post_due_chase | left_message | 1,250 |
| post_due_chase | no_answer | 1,997 |
| post_due_chase | reached_booked | 1,936 |
| post_due_chase | reached_declined | 1,121 |
| post_due_chase | wrong_number | 355 |
| pre_due_reminder | left_message | 94 |
| pre_due_reminder | no_answer | 148 |
| pre_due_reminder | reached_booked | 169 |
| pre_due_reminder | reached_declined | 84 |
| pre_due_reminder | wrong_number | 25 |
| recall_letter | letter_sent | 21,747 |

## Follow-up episodes

Episodes are reconstructed from ordinary records only. Prediction timestamp t0 = due date − 14 days; label window [t0, due + 30 days]; outcome mature when cutoff ≥ due + 37 days.

| measure | value |
|---|---|
| episodes (all) | 27,078 |
| eligible at t0 | 22,808 |
| eligible, cold start (< 180 days history) | 2,035 |
| eligible, labelled (mature, uncensored) | 22,095 |
| eligible, censored | 40 |
| eligible, immature at cutoff | 673 |
| eligible, no completion recorded by cutoff (any year) | 2,765 |
| currently open at cutoff (eligible, immature, not yet completed) — worklist candidates | 344 |

### Ineligibility reasons

| reason | count | share |
|---|---|---|
| prediction_after_cutoff | 3,099 | 0.726 |
| completed_before_prediction | 670 | 0.157 |
| transferred_before_prediction | 370 | 0.087 |
| deceased_before_prediction | 131 | 0.031 |

### Label status by programme

| programme | censored | immature | ineligible | labelled |
|---|---|---|---|---|
| htn | 15 | 286 | 2,397 | 8,422 |
| t2d | 25 | 387 | 1,873 | 13,673 |

## Label prevalence (labelled episodes)

Overall prevalence of *follow-up incomplete at due + 30 days*: **31.4%** (n = 22,095). Configured plausibility band 15.0%–35.0%: **PASS**.

| programme | labelled | prevalence |
|---|---|---|
| htn | 8,422 | 0.338 |
| t2d | 13,673 | 0.300 |

| prediction_year | labelled | prevalence |
|---|---|---|
| 2020 | 2,401 | 0.384 |
| 2021 | 5,650 | 0.335 |
| 2022 | 5,054 | 0.299 |
| 2023 | 4,974 | 0.296 |
| 2024 | 4,016 | 0.287 |

## Outcome maturity (eligible episodes by prediction year)

| prediction_year | censored | immature | labelled |
|---|---|---|---|
| 2020 | 6 | 0 | 2,401 |
| 2021 | 12 | 0 | 5,650 |
| 2022 | 6 | 0 | 5,054 |
| 2023 | 7 | 0 | 4,974 |
| 2024 | 9 | 673 | 4,016 |

## Missingness

### Null share in record columns (columns with any nulls)

| table | column | null_share |
|---|---|---|
| patients | death_date | 0.964 |
| patients | death_recorded_at | 0.964 |
| patients | transfer_out_date | 0.894 |
| patients | transfer_recorded_at | 0.894 |
| encounters | appointment_id | 0.417 |
| medication_statements | end_date | 0.943 |
| medication_statements | end_recorded_at | 0.943 |
| outreach_attempts | next_action_due | 0.073 |

Most nulls are structural (e.g. `death_date` for living patients, `appointment_id` for walk-in acute encounters, `end_date` for ongoing medicines).

### Information available at prediction time (eligible episodes)

Computed only from records with `recorded_at` (or `status_recorded_at` / `booked_at`) strictly before t0.

| measure | share of eligible episodes |
|---|---|
| T2D episodes: HbA1c recorded in prior 365 days | 100.0% |
| All episodes: systolic BP recorded in prior 365 days | 96.3% |
| All episodes: no-show recorded in prior 365 days | 12.7% |
| All episodes: outreach attempt recorded in prior 365 days | 28.0% |
| All episodes: review appointment already booked for [t0, due] at t0 | 42.6% |

## Separation of information classes

- `records/` — ordinary records as known at the cutoff (future model inputs).
- `cohort/` — episode definitions and eligibility, known at t0.
- `outcomes/` — completion, censoring, maturity and labels (targets only, never features).
- `hidden/` — latent simulation traits and the simulator's episode log (validation only).
