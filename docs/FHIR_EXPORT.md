# FHIR R4-oriented export

`GET /api/v1/patients/{patient_id}/fhir` (coordinator, admin; audited as `fhir.export`), or
`cura-api fhir-export <patient_id>`, returns one synthetic patient as a FHIR **Bundle** of type
`collection` (`application/fhir+json`).

| Resource | Source | Notes |
|---|---|---|
| Patient | patients | synthetic MRN identifier, name, gender, birthDate, deceasedDateTime, language |
| Condition | conditions | SNOMED CT code, clinical/verification status, category problem-list-item |
| Encounter | encounters | status finished, class AMB, period, link to Appointment |
| Appointment | appointments | status booked/fulfilled/noshow/cancelled; `end` is a nominal 20-minute slot (stated in `comment`) because slot length is not recorded |
| Observation | observations | LOINC code, `valueQuantity` with UCUM system and the recorded unit, effective/issued times, category laboratory or vital-signs |
| MedicationStatement | medication_statements | RxNorm code, active/stopped, effectivePeriod, dateAsserted |
| RiskAssessment | risk_scores (latest run) | probabilityDecimal, qualitativeRisk = tier, method = model version, occurrence = prediction time, whenPeriod = outcome window, note = "SIMULATED PREDICTION…" with scoring run, scored time, data fingerprint and score status |
| Communication | outreach_attempts | completed / not-done (+statusReason), medium (v3-ParticipationMode where a code applies), sent, payload with outcome, note and next action |

Every resource carries `meta.tag` `urn:cura-ai:tags#synthetic`. Entries use deterministic
`urn:uuid:` fullUrls, and every reference points to an entry inside the Bundle.

## What has been validated
- **Automated structural checks** (`cura_api.services.fhir.structural_problems`, run in tests for
  several patients):
  - Bundle type;
  - unique `urn:uuid` fullUrls;
  - FHIR id syntax;
  - every reference resolves inside the Bundle;
  - UCUM-coded quantities;
  - Appointment invariants app-2/app-3 and required participant;
  - RiskAssessment probability in 0–1 (ras-2).
- **Model parsing with `fhir.resources` 8.x R4B classes** (`r4b_parse_errors`) for every resource
  and the Bundle. R4B is the closest release that library ships; it is not identical to R4.

## What has NOT been validated
The official **HL7 FHIR Validator** was not run. It needs a separate Java download and its package
cache, and it was not available in the build environment. **Formal FHIR R4 conformance is
therefore not claimed.** To validate yourself (Java 17+ is installed on this machine):
```
.venv\Scripts\cura-api fhir-export pt_000370 --out bundle.json
curl -L -o validator_cli.jar https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar
java -jar validator_cli.jar bundle.json -version 4.0.1
```
Expect informational messages about terminology for the custom `urn:cura-ai:*` systems. Report any
errors and fix them in `backend/src/cura_api/services/fhir.py`.
