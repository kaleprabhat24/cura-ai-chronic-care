# Cura AI follow-up risk model — cura-fu-1.0.0

> **Synthetic-data evaluation, not clinical validation.** All patients, events and outcomes are simulated. These numbers describe how well the models recover structure that the simulator put into the data. They are not evidence of real-world performance or clinical benefit.

## Summary

- Selected model: **Logistic regression** (platt calibration). Selection rule (validation calibration half only): choose XGBoost only if the paired patient-level bootstrap 95% CI of its validation PR-AUC advantage over logistic regression lies entirely above 0.0 and its validation Brier score is not higher by more than 0.0; otherwise keep the simpler logistic regression. Observed validation difference XGBoost − LR: PR-AUC +0.0136 [-0.0007, 0.0284], Brier -0.0002.
- Test (held-out patients, 2024): PR-AUC 0.647, ROC-AUC 0.791, Brier 0.158, precision at 15% capacity 0.735 vs prevalence 0.291.
- Test patients overlapping training/validation patients: 0.

## Evaluation design and splits

Patients are split 80/20 at random (seed 20251016); the 20% held-out patients are used only for the final test. Train: [2020, 2021, 2022]. Validation: 2023, split by patient into a tuning half (hyper-parameters, early stopping) and a calibration half (calibration, model selection, tier thresholds). Test: 2024, held-out patients. Training labels matured before the validation year began; validation labels before the test year began.

| split | episodes | patients | prevalence |
|---|---|---|---|
| train | 11,086 | 3,304 | 0.308 |
| val_tune | 1,606 | 1,121 | 0.329 |
| val_cal | 1,546 | 1,074 | 0.285 |
| test | 777 | 526 | 0.291 |

## Label prevalence over time (all labelled episodes)

| year | labelled | all | t2d | htn |
|---|---|---|---|---|
| 2020 | 4,917 | 0.301 | 0.273 | 0.342 |
| 2021 | 4,911 | 0.305 | 0.288 | 0.329 |
| 2022 | 4,708 | 0.306 | 0.310 | 0.299 |
| 2023 | 4,519 | 0.308 | 0.297 | 0.325 |
| 2024 | 3,803 | 0.306 | 0.291 | 0.327 |

## Model comparison — test set (held-out patients)

95% intervals from a patient-level bootstrap (500 resamples). Ranking metrics use raw model scores (the worklist order); Brier/calibration use calibrated probabilities. Capacity = top 15% of episodes.

| model | PR-AUC | ROC-AUC | Brier | precision@cap | recall@cap | ECE | cal. slope |
|---|---|---|---|---|---|---|---|
| Prevalence baseline | 0.291 [0.257, 0.325] | 0.500 [0.500, 0.500] | 0.207 [0.194, 0.220] | 0.291 [0.207, 0.376] | 0.150 [0.112, 0.192] | 0.017 | – |
| Attendance heuristic | 0.398 [0.343, 0.455] | 0.611 [0.567, 0.652] | 0.201 [0.185, 0.215] | 0.479 [0.373, 0.564] | 0.248 [0.200, 0.285] | 0.064 | 1.30 |
| Logistic regression **(selected)** | 0.647 [0.588, 0.702] | 0.791 [0.751, 0.821] | 0.158 [0.143, 0.175] | 0.735 [0.641, 0.808] | 0.381 [0.336, 0.417] | 0.049 | 1.09 |
| XGBoost | 0.642 [0.580, 0.696] | 0.783 [0.744, 0.816] | 0.160 [0.146, 0.177] | 0.718 [0.628, 0.784] | 0.372 [0.324, 0.409] | 0.048 | 1.06 |

### Paired differences (test, bootstrap)

| comparison | Δ PR-AUC [95% CI] | Δ ROC-AUC [95% CI] | Δ Brier [95% CI] | Δ precision@cap [95% CI] |
|---|---|---|---|---|
| xgboost - logistic regression | [-0.028, 0.014] | [-0.018, 0.003] | [-0.0006, 0.0051] | [-0.065, 0.018] |
| xgboost - attendance heuristic | [0.180, 0.298] | [0.129, 0.218] | [-0.0517, -0.0276] | [0.146, 0.341] |
| logistic regression - attendance heuristic | [0.187, 0.305] | [0.138, 0.228] | [-0.0529, -0.0302] | [0.168, 0.362] |
| logistic regression - prevalence baseline | [0.303, 0.404] | [0.251, 0.321] | [-0.0589, -0.0368] | [0.326, 0.544] |

### Precision / recall / lift at other capacities (test)

| model | capacity | k | precision | recall | lift |
|---|---|---|---|---|---|
| Prevalence baseline | 10% | 78 | 0.308 | 0.106 | 1.06 |
| Prevalence baseline | 15% | 117 | 0.291 | 0.150 | 1.00 |
| Prevalence baseline | 20% | 156 | 0.327 | 0.226 | 1.12 |
| Attendance heuristic | 10% | 78 | 0.487 | 0.168 | 1.67 |
| Attendance heuristic | 15% | 117 | 0.479 | 0.248 | 1.65 |
| Attendance heuristic | 20% | 156 | 0.410 | 0.283 | 1.41 |
| Logistic regression | 10% | 78 | 0.795 | 0.274 | 2.73 |
| Logistic regression | 15% | 117 | 0.735 | 0.381 | 2.53 |
| Logistic regression | 20% | 156 | 0.647 | 0.447 | 2.23 |
| XGBoost | 10% | 78 | 0.756 | 0.261 | 2.60 |
| XGBoost | 15% | 117 | 0.718 | 0.372 | 2.47 |
| XGBoost | 20% | 156 | 0.679 | 0.469 | 2.34 |

## Validation results (calibration half — used for selection)

| model | PR-AUC | ROC-AUC | Brier | ECE | calibration |
|---|---|---|---|---|---|
| Prevalence baseline | 0.285 | 0.500 | 0.204 | 0.023 | identity |
| Attendance heuristic | 0.344 | 0.583 | 0.200 | 0.026 | platt |
| Logistic regression | 0.556 | 0.778 | 0.163 | 0.041 | platt |
| XGBoost | 0.570 | 0.777 | 0.163 | 0.025 | platt |

Tuning on the validation tuning half: logistic regression C = 0.003 (grid PR-AUC: C=0.003: 0.654, C=0.01: 0.653, C=0.03: 0.653, C=0.1: 0.653, C=0.3: 0.653, C=1.0: 0.653); XGBoost max_depth = 2, trees = 85 (grid: depth 2: 0.652 @ 85 trees, depth 3: 0.649 @ 107 trees, depth 4: 0.645 @ 50 trees). Calibration methods chosen by patient-grouped 5-fold CV Brier on the calibration half: Attendance heuristic platt 0.2009, isotonic 0.2016; Logistic regression platt 0.1641, isotonic 0.1642; XGBoost platt 0.1638, isotonic 0.1645.

## Calibration — Logistic regression on the test set

ECE 0.049, calibration slope 1.09, intercept 0.19, calibration-in-the-large (observed − mean predicted) +0.018.

| decile | n | mean predicted | observed rate |
|---|---|---|---|
| 1 | 78 | 0.071 | 0.077 |
| 2 | 78 | 0.099 | 0.103 |
| 3 | 78 | 0.124 | 0.064 |
| 4 | 78 | 0.144 | 0.103 |
| 5 | 78 | 0.171 | 0.231 |
| 6 | 78 | 0.207 | 0.269 |
| 7 | 78 | 0.268 | 0.321 |
| 8 | 77 | 0.392 | 0.442 |
| 9 | 77 | 0.554 | 0.506 |
| 10 | 77 | 0.705 | 0.805 |

Risk-tier thresholds (fixed on validation): High ≥ 0.569, Medium ≥ 0.241.

| tier | test episodes | share | observed incomplete rate |
|---|---|---|---|
| high | 104 | 0.134 | 0.731 |
| medium | 188 | 0.242 | 0.415 |
| low | 485 | 0.624 | 0.148 |

## Feature importance (test set, mean |contribution|, log-odds)

Feature and missing-indicator contributions are combined. Values are model-specific and not comparable as percentages of risk.

| feature | label | LR mean |contribution| | XGBoost mean |SHAP| |
|---|---|---|---|
| `has_open_booking` | Review appointment already booked for this follow-up | 0.699 | 0.126 |
| `days_since_last_outreach` | Days since last outreach attempt | 0.130 | 0.111 |
| `prior_ontime_rate_smoothed` | Earlier follow-ups completed on time, smoothed | 0.128 | 0.124 |
| `prior_episodes_matured` | Earlier follow-ups with a known outcome | 0.120 | 0.008 |
| `booking_lead_days` | Booking lead time of the open booking | 0.102 | 0.165 |
| `deprivation_quintile` | Area deprivation quintile (5 = most deprived) | 0.081 | 0.055 |
| `booking_offset_from_due_days` | Booked slot relative to due date | 0.079 | 0.623 |
| `hba1c_last` | Latest HbA1c | 0.079 | 0.056 |
| `prior_appointments_24m` | Review appointments with a known outcome (24 mo) | 0.062 | 0.000 |
| `programme_is_htn` | Follow-up is a hypertension review | 0.058 | 0.000 |
| `sbp_count_24m` | Blood pressure readings (24 mo) | 0.056 | 0.002 |
| `hba1c_count_24m` | HbA1c results (24 mo) | 0.054 | 0.045 |

XGBoost top features: `booking_offset_from_due_days`, `booking_lead_days`, `has_open_booking`, `prior_ontime_rate_smoothed`, `days_since_last_outreach`, `hba1c_last`, `deprivation_quintile`, `attendance_rate_24m_smoothed`. SHAP check: max |shap − XGBoost native contributions| = 0.00e+00; additivity error 1.12e-06.

## Example explanation (highest-scored test episode)

Episode `ep_000831_t2d_20230901` — calibrated probability 0.884 (tier high); model log-odds +2.040 = base -0.961 + contributions. Contributions are log-odds on the uncalibrated model scale (positive raises, negative lowers the model's risk estimate). They are not percentages of risk probability.

| factor | recorded value | contribution (log-odds) | direction |
|---|---|---|---|
| Review appointment already booked for this follow-up | no | +1.268 | raises the model's risk estimate |
| Latest HbA1c | 11.2 % | +0.444 | raises the model's risk estimate |
| Earlier follow-ups completed on time, smoothed | 0.47 | +0.263 | raises the model's risk estimate |
| Recall letters sent (12 mo) | 6 | +0.236 | raises the model's risk estimate |
| Consecutive most recent missed appointments | 1 | +0.225 | raises the model's risk estimate |
| Most recent known follow-up was late | yes | -0.110 | lowers the model's risk estimate |

Explanations show which recorded data the model relied on for this prediction. They are statistical associations in synthetic data, not causes, and not clinical advice.

## Leakage checks

Single-feature discrimination on the training set (max(AUC, 1 − AUC)); features ≥ 0.9 would be investigated as possible leaks. Flagged: none.

| feature | single-feature AUC |
|---|---|
| `has_open_booking` | 0.721 |
| `prior_ontime_rate_smoothed` | 0.607 |
| `attendance_rate_24m_smoothed` | 0.588 |
| `hba1c_count_24m` | 0.580 |
| `prior_episodes_matured` | 0.574 |
| `days_since_last_outreach__missing` | 0.569 |
| `sbp_count_24m` | 0.566 |
| `hba1c_trend_per_year__missing` | 0.560 |

## Missing data (share of episodes with the feature missing)

| feature | train | test |
|---|---|---|
| `booking_lead_days` | 0.282 | 0.260 |
| `booking_offset_from_due_days` | 0.282 | 0.260 |
| `last_prior_followup_late` | 0.060 | 0.015 |
| `hba1c_last` | 0.276 | 0.264 |
| `days_since_hba1c` | 0.276 | 0.264 |
| `hba1c_trend_per_year` | 0.442 | 0.438 |
| `sbp_last` | 0.007 | 0.005 |
| `days_since_sbp` | 0.007 | 0.005 |
| `sbp_trend_per_year` | 0.154 | 0.145 |
| `days_since_last_outreach` | 0.459 | 0.391 |

HbA1c features are missing by design for hypertension-only patients, and for diabetes patients whose HbA1c was not drawn or not resulted. Booking features are missing when no review is booked.

## Subgroups — Logistic regression, test set

| group | value | n | prevalence | PR-AUC | mean predicted |
|---|---|---|---|---|---|
| programme | htn | 305 | 0.311 | 0.657 | 0.293 |
| programme | t2d | 472 | 0.278 | 0.642 | 0.259 |
| sex | female | 383 | 0.282 | 0.613 | 0.254 |
| sex | male | 394 | 0.299 | 0.674 | 0.291 |
| deprivation_group | 1-2 | 327 | 0.278 | 0.634 | 0.251 |
| deprivation_group | 3 | 133 | 0.256 | 0.671 | 0.269 |
| deprivation_group | 4-5 | 317 | 0.319 | 0.656 | 0.296 |

Subgroup samples are small; differences are descriptive only.

## Limitations

- Synthetic data: the simulator determines every relationship, so the results show methodology, not clinical validity.
- Labels reflect the simulated historical outreach policy; predictions are not causal estimates of the effect of outreach.
- The test set is one year of held-out patients (n = 777); intervals are wide for small subgroups.
- Explanations are model contributions (log-odds), not causes and not risk percentages.
- Feature spec fs-1.0; any change to features or data requires retraining and re-evaluation.
