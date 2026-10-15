"""Code catalogs used by the generator.

Codes are standard identifiers chosen for illustration (SNOMED CT, LOINC, RxNorm, UCUM units).
They should be re-verified against the official terminologies before any real interoperability use.
"""

from __future__ import annotations

SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"

CONDITIONS = {
    "t2d": {"code": "44054006", "display": "Diabetes mellitus type 2"},
    "htn": {"code": "38341003", "display": "Hypertensive disorder, systemic arterial"},
}

# key -> (LOINC code, display, UCUM unit, plausible (min, max))
OBSERVATIONS = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%", (4.0, 16.0)),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]", (80.0, 230.0)),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]", (40.0, 140.0)),
    "weight": ("29463-7", "Body weight", "kg", (35.0, 220.0)),
    "heart_rate": ("8867-4", "Heart rate", "/min", (35.0, 180.0)),
}
UNIT_BY_LOINC = {code: unit for code, _, unit, _ in OBSERVATIONS.values()}
RANGE_BY_LOINC = {code: rng for code, _, _, rng in OBSERVATIONS.values()}

# key -> (RxNorm code, display, drug class)
MEDICATIONS = {
    "metformin": ("6809", "metformin", "biguanide"),
    "glipizide": ("4821", "glipizide", "sulfonylurea"),
    "empagliflozin": ("1545653", "empagliflozin", "sglt2_inhibitor"),
    "insulin_glargine": ("274783", "insulin glargine", "basal_insulin"),
    "atorvastatin": ("83367", "atorvastatin", "statin"),
    "lisinopril": ("29046", "lisinopril", "ace_inhibitor"),
    "amlodipine": ("17767", "amlodipine", "calcium_channel_blocker"),
    "hydrochlorothiazide": ("5487", "hydrochlorothiazide", "thiazide_diuretic"),
}
# Treatment-intensification ladders (first inactive drug is added next).
LADDERS = {
    "t2d": ["metformin", "glipizide", "empagliflozin", "insulin_glargine"],
    "htn": ["lisinopril", "amlodipine", "hydrochlorothiazide"],
}

APPOINTMENT_STATUSES = ("booked", "attended", "no_show", "cancelled_patient", "cancelled_clinic")
ENCOUNTER_TYPES = ("review_t2d", "review_htn", "acute")
OUTREACH_OUTCOMES = (
    "reached_booked", "reached_declined", "no_answer", "left_message", "wrong_number",
    "letter_sent",
)

# Fictional name parts. Combinations are random and do not represent real people.
GIVEN_NAMES = {
    "female": [
        "Amara", "Beatrice", "Chloe", "Daria", "Elena", "Farah", "Grace", "Hana", "Ines", "Joy",
        "Kavya", "Lena", "Maya", "Nadia", "Olive", "Priya", "Rosa", "Sana", "Tara", "Zofia",
    ],
    "male": [
        "Aaron", "Bilal", "Callum", "Dev", "Emeka", "Felix", "Gabriel", "Hamza", "Ivan", "Jonah",
        "Kofi", "Luca", "Mateo", "Nikhil", "Oscar", "Pavel", "Rahul", "Samuel", "Tomas", "Yusuf",
    ],
}
FAMILY_NAMES = [
    "Ashdown", "Bramley", "Carrow", "Delacey", "Ellery", "Fenwick", "Garside", "Hollis",
    "Ingram", "Jessop", "Kendrew", "Larkin", "Merrow", "Northam", "Ottley", "Pendle", "Quarry",
    "Redfern", "Stanbury", "Thorne", "Underhill", "Vance", "Whitlock", "Yardley",
]
