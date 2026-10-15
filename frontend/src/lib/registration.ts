/** Patient-registration form model, validation and payload building (pure; unit-tested).
 * Client-side checks mirror the API's rules for fast feedback; the API remains authoritative. */

import type { Schemas } from "@/lib/api/client";

export const SYNTHETIC_WARNING =
  "Synthetic demonstration environment — enter fictional patient information only. Do not enter real patient data.";

export type Programme = "t2d" | "htn";
export const PROGRAMMES: { id: Programme; label: string; intervalDays: number; reviewType: string }[] = [
  { id: "t2d", label: "Type 2 diabetes", intervalDays: 182, reviewType: "review_t2d" },
  { id: "htn", label: "Hypertension", intervalDays: 365, reviewType: "review_htn" },
];
export const LEAD_DAYS = 14;

export const OBSERVATION_KINDS = {
  hba1c: { label: "HbA1c", unit: "%", min: 4, max: 16, step: 0.1 },
  sbp: { label: "Systolic blood pressure", unit: "mm[Hg]", min: 80, max: 230, step: 1 },
  dbp: { label: "Diastolic blood pressure", unit: "mm[Hg]", min: 40, max: 140, step: 1 },
  weight: { label: "Body weight", unit: "kg", min: 35, max: 220, step: 0.1 },
  heart_rate: { label: "Heart rate", unit: "/min", min: 35, max: 180, step: 1 },
} as const;
export type ObservationKind = keyof typeof OBSERVATION_KINDS;

export const MEDICATIONS = {
  metformin: "Metformin",
  glipizide: "Glipizide",
  empagliflozin: "Empagliflozin",
  insulin_glargine: "Insulin glargine",
  atorvastatin: "Atorvastatin",
  lisinopril: "Lisinopril",
  amlodipine: "Amlodipine",
  hydrochlorothiazide: "Hydrochlorothiazide",
} as const;
export type MedicationKey = keyof typeof MEDICATIONS;

export const ENCOUNTER_TYPES = {
  review_t2d: "Diabetes review",
  review_htn: "Hypertension review",
  acute: "Unplanned (acute) visit",
} as const;
export type EncounterType = keyof typeof ENCOUNTER_TYPES;

export const APPOINTMENT_STATUSES = {
  attended: "Attended",
  no_show: "Did not attend",
  cancelled_patient: "Cancelled by patient",
  cancelled_clinic: "Cancelled by clinic",
  booked: "Booked (future)",
} as const;
export type AppointmentStatus = keyof typeof APPOINTMENT_STATUSES;

export type AppointmentRow = { programme: Programme; scheduledFor: string; status: AppointmentStatus };
export type EncounterRow = { type: EncounterType; startAt: string; duration: string };
export type ObservationRow = { kind: ObservationKind; value: string; effectiveAt: string };
export type MedicationRow = { medication: MedicationKey; startDate: string; endDate: string };
export type OutreachRow = { programme: Programme; channel: string; outcome: string; attemptedAt: string; note: string };

export type HistoryDraft = {
  appointments: AppointmentRow[];
  encounters: EncounterRow[];
  observations: ObservationRow[];
  medications: MedicationRow[];
};

export type RegistrationDraft = HistoryDraft & {
  givenName: string;
  familyName: string;
  birthDate: string;
  sex: "" | "female" | "male";
  deprivation: string;
  language: string;
  registeredOn: string;
  conditions: Record<Programme, { enabled: boolean; onsetDate: string }>;
  outreach: OutreachRow[];
};

export const emptyHistory = (): HistoryDraft => ({ appointments: [], encounters: [], observations: [], medications: [] });

export const emptyDraft = (): RegistrationDraft => ({
  givenName: "",
  familyName: "",
  birthDate: "",
  sex: "",
  deprivation: "",
  language: "en",
  registeredOn: "",
  conditions: { t2d: { enabled: false, onsetDate: "" }, htn: { enabled: false, onsetDate: "" } },
  outreach: [],
  ...emptyHistory(),
});

const NAME = /^[A-Za-z][A-Za-z '\-.]{0,79}$/;
const SLACK_MS = 5 * 60 * 1000;

/** `datetime-local` value (browser local time) -> ISO-8601 UTC, or null if invalid. */
export function localToIso(value: string): string | null {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

function parseDate(value: string): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const d = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function todayIso(now: Date): string {
  return now.toISOString().slice(0, 10);
}

export type Errors = Record<string, string>;

export function validateHistory(h: HistoryDraft, now: Date, programmes: Set<Programme>, prefix = "", birthDate?: string): Errors {
  const e: Errors = {};
  const birth = birthDate ? parseDate(birthDate) : null;
  const tooEarly = (d: Date) => birth !== null && d < birth;
  h.appointments.forEach((a, i) => {
    const key = `${prefix}appointments.${i}`;
    const iso = localToIso(a.scheduledFor);
    if (!iso) e[key] = "Enter the appointment date and time.";
    else {
      const future = new Date(iso).getTime() > now.getTime() + SLACK_MS;
      if (future && (a.status === "attended" || a.status === "no_show")) e[key] = "A future appointment cannot have an attendance result.";
      else if (!future && a.status === "booked") e[key] = "A past appointment needs its outcome.";
      else if (!programmes.has(a.programme)) e[key] = "Register that condition first.";
      else if (tooEarly(new Date(iso))) e[key] = "Cannot be before the date of birth.";
    }
  });
  h.encounters.forEach((x, i) => {
    const key = `${prefix}encounters.${i}`;
    const iso = localToIso(x.startAt);
    const minutes = Number(x.duration || "20");
    if (!iso) e[key] = "Enter the visit date and time.";
    else if (new Date(iso).getTime() > now.getTime() + SLACK_MS) e[key] = "A completed visit cannot be in the future.";
    else if (x.type !== "acute" && !programmes.has(x.type.replace("review_", "") as Programme)) e[key] = "Register that condition first.";
    else if (!Number.isInteger(minutes) || minutes < 5 || minutes > 240) e[key] = "Duration must be 5–240 minutes.";
    else if (tooEarly(new Date(iso))) e[key] = "Cannot be before the date of birth.";
  });
  h.observations.forEach((o, i) => {
    const key = `${prefix}observations.${i}`;
    const spec = OBSERVATION_KINDS[o.kind];
    const value = Number(o.value);
    const iso = localToIso(o.effectiveAt);
    if (o.value.trim() === "" || Number.isNaN(value)) e[key] = `Enter a number in ${spec.unit}.`;
    else if (value < spec.min || value > spec.max) e[key] = `${spec.label} must be ${spec.min}–${spec.max} ${spec.unit}.`;
    else if (!iso) e[key] = "Enter when it was measured.";
    else if (new Date(iso).getTime() > now.getTime() + SLACK_MS) e[key] = "A measurement cannot be in the future.";
    else if (tooEarly(new Date(iso))) e[key] = "Cannot be before the date of birth.";
  });
  h.medications.forEach((x, i) => {
    const key = `${prefix}medications.${i}`;
    const start = parseDate(x.startDate);
    const end = x.endDate ? parseDate(x.endDate) : null;
    if (!start) e[key] = "Enter the start date.";
    else if (x.startDate > todayIso(now)) e[key] = "The start date cannot be in the future.";
    else if (x.endDate && (!end || x.endDate < x.startDate || x.endDate > todayIso(now))) e[key] = "The stop date must be between the start date and today.";
    else if (tooEarly(start)) e[key] = "Cannot be before the date of birth.";
  });
  return e;
}

export function validateDraft(d: RegistrationDraft, now: Date = new Date()): Errors {
  const e: Errors = {};
  if (!NAME.test(d.givenName.trim())) e.givenName = "Use letters, spaces, hyphens or apostrophes.";
  if (!NAME.test(d.familyName.trim())) e.familyName = "Use letters, spaces, hyphens or apostrophes.";
  const birth = parseDate(d.birthDate);
  if (!birth) e.birthDate = "Enter the date of birth.";
  else {
    const t = now;
    let age = t.getUTCFullYear() - birth.getUTCFullYear();
    if (t.getUTCMonth() < birth.getUTCMonth() || (t.getUTCMonth() === birth.getUTCMonth() && t.getUTCDate() < birth.getUTCDate())) age -= 1;
    if (age < 18 || age > 110) e.birthDate = "Patients must be aged 18–110.";
  }
  if (d.sex !== "female" && d.sex !== "male") e.sex = "Choose the recorded sex.";
  const q = Number(d.deprivation);
  if (!Number.isInteger(q) || q < 1 || q > 5) e.deprivation = "Choose a quintile from 1 to 5.";
  if (!/^[a-z]{2,3}$/.test(d.language)) e.language = "Use a 2–3 letter language code, e.g. en.";
  const reg = parseDate(d.registeredOn);
  if (!reg) e.registeredOn = "Enter the registration date.";
  else if (d.registeredOn > todayIso(now) || (birth && reg < birth)) e.registeredOn = "Must be between the date of birth and today.";
  const programmes = new Set(PROGRAMMES.filter((p) => d.conditions[p.id].enabled).map((p) => p.id));
  if (programmes.size === 0) e.conditions = "Select at least one chronic condition.";
  for (const p of programmes) {
    const onset = parseDate(d.conditions[p].onsetDate);
    if (!onset) e[`conditions.${p}`] = "Enter the diagnosis date.";
    else if (d.conditions[p].onsetDate > todayIso(now) || (birth && onset < birth)) e[`conditions.${p}`] = "Must be between the date of birth and today.";
  }
  Object.assign(e, validateHistory(d, now, programmes, "", d.birthDate));
  d.outreach.forEach((o, i) => {
    const iso = localToIso(o.attemptedAt);
    if (!iso) e[`outreach.${i}`] = "Enter when the attempt happened.";
    else if (new Date(iso).getTime() > now.getTime() + SLACK_MS) e[`outreach.${i}`] = "Cannot be in the future.";
    else if (!programmes.has(o.programme)) e[`outreach.${i}`] = "Register that condition first.";
    else if (!o.outcome) e[`outreach.${i}`] = "Choose the outcome.";
    else if (o.note.length > 2000) e[`outreach.${i}`] = "Keep the note under 2000 characters.";
  });
  return e;
}

export function historyPayload(h: HistoryDraft): Schemas["HistoryIn"] {
  return {
    appointments: h.appointments.map((a) => ({ programme: a.programme, scheduled_for: localToIso(a.scheduledFor)!, status: a.status })),
    encounters: h.encounters.map((x) => ({ encounter_type: x.type, start_at: localToIso(x.startAt)!, duration_minutes: Number(x.duration || "20") })),
    observations: h.observations.map((o) => ({ kind: o.kind, value: Number(o.value), unit: OBSERVATION_KINDS[o.kind].unit, effective_at: localToIso(o.effectiveAt)! })),
    medications: h.medications.map((x) => ({ medication: x.medication, start_date: x.startDate, end_date: x.endDate || null })),
  };
}

type RegistrationBody = Schemas["PatientRegistration"] & { history: NonNullable<Schemas["PatientRegistration"]["history"]> };

export function registrationPayload(d: RegistrationDraft, submissionId: string, allowDuplicate = false): RegistrationBody {
  return {
    submission_id: submissionId,
    given_name: d.givenName.trim(),
    family_name: d.familyName.trim(),
    birth_date: d.birthDate,
    sex: d.sex as "female" | "male",
    deprivation_quintile: Number(d.deprivation),
    preferred_language: d.language,
    registered_on: d.registeredOn,
    conditions: PROGRAMMES.filter((p) => d.conditions[p.id].enabled).map((p) => ({ programme: p.id, onset_date: d.conditions[p.id].onsetDate })),
    history: {
      ...historyPayload(d),
      outreach: d.outreach.map((o) => ({
        programme: o.programme,
        channel: o.channel as Schemas["OutreachHistoryIn"]["channel"],
        outcome: o.outcome as Schemas["OutreachHistoryIn"]["outcome"],
        attempted_at: localToIso(o.attemptedAt)!,
        note: o.note.trim() || null,
        next_action: "none",
        next_action_due: null,
      })),
    },
    allow_possible_duplicate: allowDuplicate,
  };
}

/** Preview of the follow-up the server will derive: the latest completed review (review visit or
 * attended review appointment) + the programme interval; prediction time = due − 14 days. */
export function followUpPreview(d: HistoryDraft, programme: Programme): { lastReview: string; due: string; predictionAt: string } | null {
  const p = PROGRAMMES.find((x) => x.id === programme)!;
  const times = [
    ...d.encounters.filter((x) => x.type === p.reviewType).map((x) => localToIso(x.startAt)),
    ...d.appointments.filter((a) => a.programme === programme && a.status === "attended").map((a) => localToIso(a.scheduledFor)),
  ].filter((x): x is string => !!x);
  if (!times.length) return null;
  const last = times.sort().at(-1)!;
  const lastDay = new Date(`${last.slice(0, 10)}T00:00:00Z`);
  const due = new Date(lastDay.getTime() + p.intervalDays * 86400000);
  const t0 = new Date(due.getTime() - LEAD_DAYS * 86400000);
  return { lastReview: last.slice(0, 10), due: due.toISOString().slice(0, 10), predictionAt: t0.toISOString().slice(0, 10) };
}

export function historyCount(h: HistoryDraft & { outreach?: OutreachRow[] }): number {
  return h.appointments.length + h.encounters.length + h.observations.length + h.medications.length + (h.outreach?.length ?? 0);
}

export function newSubmissionId(): string {
  return globalThis.crypto.randomUUID();
}
