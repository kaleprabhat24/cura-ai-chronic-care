"use client";

/* Repeatable clinical-history entry sections, shared by patient registration and "Add history". */
import { Plus, Trash2 } from "lucide-react";
import { Badge, Button, Input, Label, Select } from "@/components/ui/primitives";
import { InlineNotice } from "@/components/states";
import { CHANNELS, OUTCOME_LABELS, conditionLabel, fmtDate, humanize } from "@/lib/format";
import {
  APPOINTMENT_STATUSES,
  ENCOUNTER_TYPES,
  MEDICATIONS,
  OBSERVATION_KINDS,
  type AppointmentStatus,
  type EncounterType,
  type Errors,
  type HistoryDraft,
  type MedicationKey,
  type ObservationKind,
  type OutreachRow,
  type Programme,
} from "@/lib/registration";
import type { Schemas } from "@/lib/api/client";

const HISTORY_OUTCOMES = Object.keys(OUTCOME_LABELS) as Schemas["OutreachCreate"]["outcome"][];

function Section({ title, hint, count, onAdd, addLabel, children }: {
  title: string;
  hint: string;
  count: number;
  onAdd: () => void;
  addLabel: string;
  children: React.ReactNode;
}) {
  return (
    <fieldset className="rounded-lg border border-border p-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <legend className="flex items-center gap-2 text-[13px] font-semibold text-ink">
            {title} <Badge>{count}</Badge>
          </legend>
          <p className="text-xs text-ink-subtle">{hint}</p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={onAdd}>
          <Plus /> {addLabel}
        </Button>
      </div>
      <div className="space-y-2">{children}</div>
    </fieldset>
  );
}

function Row({ error, onRemove, children, label }: { error?: string; onRemove: () => void; children: React.ReactNode; label: string }) {
  return (
    <div className="rounded-md bg-surface-muted/50 p-2.5">
      <div className="flex flex-wrap items-end gap-2">
        {children}
        <Button type="button" variant="ghost" size="icon" onClick={onRemove} aria-label={`Remove ${label}`}>
          <Trash2 />
        </Button>
      </div>
      {error && <p className="mt-1.5 text-xs text-danger" role="alert">{error}</p>}
    </div>
  );
}

function Field({ label, id, children, className }: { label: string; id: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={className ?? "min-w-36 flex-1 space-y-1"}>
      <Label htmlFor={id} className="text-xs">{label}</Label>
      {children}
    </div>
  );
}

export function HistoryEditor({ value, onChange, programmes, errors, idPrefix = "h" }: {
  value: HistoryDraft;
  onChange: (next: HistoryDraft) => void;
  programmes: Programme[];
  errors: Errors;
  idPrefix?: string;
}) {
  const first = programmes[0] ?? "t2d";
  const set = <K extends keyof HistoryDraft>(key: K, rows: HistoryDraft[K]) => onChange({ ...value, [key]: rows });
  const update = <K extends keyof HistoryDraft>(key: K, i: number, patch: Partial<HistoryDraft[K][number]>) =>
    set(key, value[key].map((r, j) => (j === i ? { ...r, ...patch } : r)) as HistoryDraft[K]);
  const remove = <K extends keyof HistoryDraft>(key: K, i: number) =>
    set(key, value[key].filter((_, j) => j !== i) as HistoryDraft[K]);
  const encounterTypes = (Object.keys(ENCOUNTER_TYPES) as EncounterType[]).filter(
    (t) => t === "acute" || programmes.includes(t.replace("review_", "") as Programme),
  );

  return (
    <div className="space-y-3">
      <Section title="Review visits and other encounters" count={value.encounters.length} addLabel="Add visit"
               hint="Completed visits. The latest review visit sets when the next follow-up is due."
               onAdd={() => set("encounters", [...value.encounters, { type: encounterTypes[0] ?? "acute", startAt: "", duration: "20" }])}>
        {value.encounters.map((x, i) => (
          <Row key={i} label="visit" error={errors[`encounters.${i}`]} onRemove={() => remove("encounters", i)}>
            <Field label="Visit type" id={`${idPrefix}-enc-type-${i}`}>
              <Select id={`${idPrefix}-enc-type-${i}`} value={x.type} onChange={(e) => update("encounters", i, { type: e.target.value as EncounterType })}>
                {encounterTypes.map((t) => <option key={t} value={t}>{ENCOUNTER_TYPES[t]}</option>)}
              </Select>
            </Field>
            <Field label="Date and time" id={`${idPrefix}-enc-at-${i}`}>
              <Input id={`${idPrefix}-enc-at-${i}`} type="datetime-local" value={x.startAt} onChange={(e) => update("encounters", i, { startAt: e.target.value })} />
            </Field>
            <Field label="Minutes" id={`${idPrefix}-enc-min-${i}`} className="w-24 space-y-1">
              <Input id={`${idPrefix}-enc-min-${i}`} type="number" min={5} max={240} value={x.duration} onChange={(e) => update("encounters", i, { duration: e.target.value })} />
            </Field>
          </Row>
        ))}
      </Section>

      <Section title="Appointments" count={value.appointments.length} addLabel="Add appointment"
               hint="Past appointments with their outcome, or future bookings. An attended review appointment counts as a review."
               onAdd={() => set("appointments", [...value.appointments, { programme: first, scheduledFor: "", status: "attended" }])}>
        {value.appointments.map((a, i) => (
          <Row key={i} label="appointment" error={errors[`appointments.${i}`]} onRemove={() => remove("appointments", i)}>
            <Field label="Programme" id={`${idPrefix}-ap-prog-${i}`}>
              <Select id={`${idPrefix}-ap-prog-${i}`} value={a.programme} onChange={(e) => update("appointments", i, { programme: e.target.value as Programme })}>
                {programmes.map((p) => <option key={p} value={p}>{conditionLabel(p)} review</option>)}
              </Select>
            </Field>
            <Field label="Date and time" id={`${idPrefix}-ap-at-${i}`}>
              <Input id={`${idPrefix}-ap-at-${i}`} type="datetime-local" value={a.scheduledFor} onChange={(e) => update("appointments", i, { scheduledFor: e.target.value })} />
            </Field>
            <Field label="Status" id={`${idPrefix}-ap-status-${i}`}>
              <Select id={`${idPrefix}-ap-status-${i}`} value={a.status} onChange={(e) => update("appointments", i, { status: e.target.value as AppointmentStatus })}>
                {(Object.keys(APPOINTMENT_STATUSES) as AppointmentStatus[]).map((st) => <option key={st} value={st}>{APPOINTMENT_STATUSES[st]}</option>)}
              </Select>
            </Field>
          </Row>
        ))}
      </Section>

      <Section title="Observations" count={value.observations.length} addLabel="Add observation"
               hint="Measured values in the units shown. Ranges outside plausible limits are rejected."
               onAdd={() => set("observations", [...value.observations, { kind: programmes.includes("t2d") ? "hba1c" : "sbp", value: "", effectiveAt: "" }])}>
        {value.observations.map((o, i) => {
          const spec = OBSERVATION_KINDS[o.kind];
          return (
            <Row key={i} label="observation" error={errors[`observations.${i}`]} onRemove={() => remove("observations", i)}>
              <Field label="Measurement" id={`${idPrefix}-obs-kind-${i}`}>
                <Select id={`${idPrefix}-obs-kind-${i}`} value={o.kind} onChange={(e) => update("observations", i, { kind: e.target.value as ObservationKind })}>
                  {(Object.keys(OBSERVATION_KINDS) as ObservationKind[]).map((k) => <option key={k} value={k}>{OBSERVATION_KINDS[k].label}</option>)}
                </Select>
              </Field>
              <Field label={`Value (${spec.unit})`} id={`${idPrefix}-obs-val-${i}`} className="w-32 space-y-1">
                <Input id={`${idPrefix}-obs-val-${i}`} type="number" inputMode="decimal" step={spec.step} min={spec.min} max={spec.max} value={o.value}
                       onChange={(e) => update("observations", i, { value: e.target.value })} />
              </Field>
              <Field label="Measured at" id={`${idPrefix}-obs-at-${i}`}>
                <Input id={`${idPrefix}-obs-at-${i}`} type="datetime-local" value={o.effectiveAt} onChange={(e) => update("observations", i, { effectiveAt: e.target.value })} />
              </Field>
            </Row>
          );
        })}
      </Section>

      <Section title="Medications" count={value.medications.length} addLabel="Add medication"
               hint="Medicines started (and stopped, if no longer taken)."
               onAdd={() => set("medications", [...value.medications, { medication: programmes.includes("t2d") ? "metformin" : "lisinopril", startDate: "", endDate: "" }])}>
        {value.medications.map((x, i) => (
          <Row key={i} label="medication" error={errors[`medications.${i}`]} onRemove={() => remove("medications", i)}>
            <Field label="Medicine" id={`${idPrefix}-med-${i}`}>
              <Select id={`${idPrefix}-med-${i}`} value={x.medication} onChange={(e) => update("medications", i, { medication: e.target.value as MedicationKey })}>
                {(Object.keys(MEDICATIONS) as MedicationKey[]).map((k) => <option key={k} value={k}>{MEDICATIONS[k]}</option>)}
              </Select>
            </Field>
            <Field label="Started" id={`${idPrefix}-med-start-${i}`}>
              <Input id={`${idPrefix}-med-start-${i}`} type="date" value={x.startDate} onChange={(e) => update("medications", i, { startDate: e.target.value })} />
            </Field>
            <Field label="Stopped (optional)" id={`${idPrefix}-med-end-${i}`}>
              <Input id={`${idPrefix}-med-end-${i}`} type="date" value={x.endDate} onChange={(e) => update("medications", i, { endDate: e.target.value })} />
            </Field>
          </Row>
        ))}
      </Section>
    </div>
  );
}

export function OutreachHistoryEditor({ value, onChange, programmes, errors }: {
  value: OutreachRow[];
  onChange: (rows: OutreachRow[]) => void;
  programmes: Programme[];
  errors: Errors;
}) {
  const update = (i: number, patch: Partial<OutreachRow>) => onChange(value.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  return (
    <Section title="Past outreach attempts" count={value.length} addLabel="Add attempt"
             hint="Earlier contact attempts. Each is linked to the follow-up episode it belongs to, if any."
             onAdd={() => onChange([...value, { programme: programmes[0] ?? "t2d", channel: "phone", outcome: "", attemptedAt: "", note: "" }])}>
      {value.map((o, i) => (
        <Row key={i} label="outreach attempt" error={errors[`outreach.${i}`]} onRemove={() => onChange(value.filter((_, j) => j !== i))}>
          <Field label="Programme" id={`or-prog-${i}`}>
            <Select id={`or-prog-${i}`} value={o.programme} onChange={(e) => update(i, { programme: e.target.value as Programme })}>
              {programmes.map((p) => <option key={p} value={p}>{conditionLabel(p)}</option>)}
            </Select>
          </Field>
          <Field label="Channel" id={`or-ch-${i}`}>
            <Select id={`or-ch-${i}`} value={o.channel} onChange={(e) => update(i, { channel: e.target.value })}>
              {CHANNELS.map((c) => <option key={c} value={c}>{humanize(c)}</option>)}
            </Select>
          </Field>
          <Field label="Outcome" id={`or-out-${i}`}>
            <Select id={`or-out-${i}`} value={o.outcome} onChange={(e) => update(i, { outcome: e.target.value })}>
              <option value="">Choose…</option>
              {HISTORY_OUTCOMES.map((c) => <option key={c} value={c}>{OUTCOME_LABELS[c]}</option>)}
            </Select>
          </Field>
          <Field label="Attempted at" id={`or-at-${i}`}>
            <Input id={`or-at-${i}`} type="datetime-local" value={o.attemptedAt} onChange={(e) => update(i, { attemptedAt: e.target.value })} />
          </Field>
          <Field label="Note (optional)" id={`or-note-${i}`} className="w-full space-y-1">
            <Input id={`or-note-${i}`} maxLength={2000} value={o.note} onChange={(e) => update(i, { note: e.target.value })} />
          </Field>
        </Row>
      ))}
    </Section>
  );
}

const STATUS_TONE = { scored: "ok", already_scored: "ok", ready_to_score: "info", unavailable: "warn" } as const;
const STATUS_LABEL = { scored: "Scored", already_scored: "Scored", ready_to_score: "Ready to score", unavailable: "Prediction unavailable" };

/** One line per programme saying whether a prediction exists and, if not, why. */
export function ScoreStatusList({ items }: { items: Schemas["ScoreAvailability"][] }) {
  if (!items.length) return <InlineNotice tone="warn">No follow-up could be derived yet: record a review visit to start one.</InlineNotice>;
  return (
    <ul className="space-y-2">
      {items.map((s, i) => (
        <li key={`${s.programme}-${s.episode_id ?? i}`} className="rounded-lg border border-border px-3 py-2 text-[13px]">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-ink">{conditionLabel(s.programme)}</span>
            <Badge tone={STATUS_TONE[s.status]}>{STATUS_LABEL[s.status]}</Badge>
            {s.due_date && <span className="text-xs text-ink-subtle">due {fmtDate(s.due_date)}</span>}
            {s.prediction_at && <span className="text-xs text-ink-subtle">· prediction time {fmtDate(s.prediction_at)}</span>}
          </div>
          <p className="mt-1 text-xs text-ink-muted">{s.message}</p>
        </li>
      ))}
    </ul>
  );
}
