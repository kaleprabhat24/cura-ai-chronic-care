/** Display formatting. Timestamps from the API are ISO-8601 UTC; dates are shown in UTC so they match
 * the dataset's recorded values exactly. */

const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const dateTimeFmt = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  timeZone: "UTC",
});
const monthFmt = new Intl.DateTimeFormat("en-GB", { month: "short", year: "2-digit", timeZone: "UTC" });

export function fmtDate(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value.length === 10 ? `${value}T00:00:00Z` : value);
  return Number.isNaN(d.getTime()) ? value : dateFmt.format(d);
}

export function fmtDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : `${dateTimeFmt.format(d)} UTC`;
}

export function fmtMonth(value: string): string {
  const d = new Date(`${value}-01T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? value : monthFmt.format(d);
}

export function pct(value: number | null | undefined, digits = 0): string {
  return value == null || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 0): string {
  return value == null || Number.isNaN(value)
    ? "—"
    : value.toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function humanize(value: string | null | undefined): string {
  if (!value) return "—";
  const s = value.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function relativeDays(days: number): string {
  if (days === 0) return "today";
  const abs = Math.abs(days);
  const unit = abs >= 365 ? `${(abs / 365).toFixed(1)} y` : abs >= 60 ? `${Math.round(abs / 30)} mo` : `${abs} d`;
  return days > 0 ? `in ${unit}` : `${unit} ago`;
}

export const OUTCOMES = [
  "no_answer",
  "left_message",
  "reached_booked",
  "reached_declined",
  "reached_already_done",
  "wrong_number",
  "other",
] as const;
export const CHANNELS = ["phone", "sms", "letter", "email", "in_person"] as const;
export const NEXT_ACTIONS = [
  "none",
  "retry",
  "retry_later",
  "book_appointment",
  "await_response",
  "update_contact_details",
  "escalate",
  "other",
] as const;

export const OUTCOME_LABELS: Record<string, string> = {
  no_answer: "No answer",
  left_message: "Left message",
  reached_booked: "Reached — booked",
  reached_declined: "Reached — declined",
  reached_already_done: "Reached — already done",
  wrong_number: "Wrong number",
  other: "Other",
};

export function conditionLabel(programme: string): string {
  return programme === "t2d" ? "Type 2 diabetes" : programme === "htn" ? "Hypertension" : programme;
}
