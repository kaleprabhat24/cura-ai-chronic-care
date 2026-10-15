# Data dictionary (dataset v1)

All timestamps are tz-aware **UTC**; dates are calendar dates (Parquet `date32`). Every record is
fictional. `*_recorded_at` is when the information became known to the system; feature code must
filter on it.

## records/ — ordinary records as known at the data cutoff

### patients
| Column | Type | Description |
|---|---|---|
| patient_id | str | `pt_000001` … |
| synthetic_mrn | str | Fictional record number (`SYN…`) |
| given_name, family_name | str | Random fictional name parts |
| birth_date | date | |
| sex | str | `female` / `male` |
| deprivation_quintile | int | 1 (least) – 5 (most deprived), synthetic area-level proxy |
| preferred_language | str | ISO 639-1 code |
| registered_on | date | Practice registration (may precede history start) |
| death_date, death_recorded_at | date, ts | Null unless the death was recorded by the cutoff |
| transfer_out_date, transfer_recorded_at | date, ts | Null unless the transfer was recorded by the cutoff |
| is_synthetic | bool | Always true |

### conditions
`condition_id, patient_id, code_system (SNOMED CT URI), code, display, onset_date, recorded_at,
clinical_status`. Prevalent conditions may be recorded before the history start (carried-over
problem list). Incident conditions are recorded at the diagnosing review.

### appointments
| Column | Description |
|---|---|
| appointment_type | `review_t2d` / `review_htn` |
| booked_at | When the booking was made |
| scheduled_for | Slot start time (weekdays 08:30–16:50) |
| status | `attended`, `no_show`, `cancelled_patient`, `cancelled_clinic`, `booked` (still pending at the cutoff) |
| status_recorded_at | When the current status became known (`booked_at` for pending bookings) |
| booking_source | `practice` (first review), `practice_prebook`, `patient`, `outreach`, `clinic_rebook` |

### encounters
`encounter_id, patient_id, encounter_type (review_t2d | review_htn | acute), start_at, end_at,
status (finished), appointment_id (null for walk-in acute visits), recorded_at`.

### observations
| Column | Description |
|---|---|
| code_system, code, display | LOINC: 4548-4 HbA1c (`%`), 8480-6 systolic BP and 8462-4 diastolic BP (`mm[Hg]`), 29463-7 body weight (`kg`), 8867-4 heart rate (`/min`) |
| value_num, unit | Numeric value with UCUM unit (validated per code by tests) |
| effective_at | Measurement / sample time |
| recorded_at | Result availability (HbA1c lab results arrive 1–4 days later) |
| encounter_id | Encounter at which it was taken |

### medication_statements
`medication_statement_id, patient_id, code_system (RxNorm URI), code, display, drug_class, start_date,
end_date, recorded_at, end_recorded_at`. A null `end_date` means the medicine was ongoing at the
cutoff, or that its stop had not been recorded yet.

### outreach_attempts (historical, simulated practice policy)
| Column | Description |
|---|---|
| programme | `t2d` / `htn` |
| purpose | `pre_due_reminder` (7 days before t0, for recent no-show patients), `post_due_chase`, `recall_letter` |
| channel | `phone` / `letter` |
| outcome | `reached_booked`, `reached_declined`, `no_answer`, `left_message`, `wrong_number`, `letter_sent` |
| next_action, next_action_due | Planned follow-up recorded with the attempt |
| performed_by | `historical_staff` (synthetic) |
| attempted_at, recorded_at | |

## cohort/followup_episodes — known at the prediction timestamp
`episode_id, patient_id, programme, condition_code, index_event_at, index_event_type
(encounter | appointment | observation), index_event_ref, due_date, prediction_at, window_end_date,
eligible, ineligible_reason, is_cold_start, history_days`. See `PREDICTION_TARGET.md`.

## outcomes/episode_labels — future relative to t0 (targets only)
`episode_id, completed_at, completion_event_type, completion_event_ref, completed_in_window,
censored, censor_reason (death | transfer_out), censor_date, mature,
label_status (ineligible | censored | immature | labelled), label (1 = follow-up incomplete at
due + 30 days, 0 = completed; null unless labelled)`.

## hidden/ — never ML inputs
- `simulation_traits`: `engagement`, `access_barrier`, `instability_entry_prob`, `trust`,
  `unstable_month_fraction`, `hba1c_baseline`, `sbp_baseline` (latent drivers of the simulation).
- `simulator_episode_log`: the simulator's own index/due/completion dates, used only to test that the
  record-based episode reconstruction matches the ground truth.

## manifest.json
Seed, config SHA-256 and full config, start and cutoff dates, and per file: row count, column list,
`content_sha256` (hash of the data as read back) and `file_sha256` (hash of the bytes). Also records
library versions.
