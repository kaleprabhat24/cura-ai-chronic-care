"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ArrowLeft, CheckCircle2, ClipboardCheck, Pencil, UserPlus } from "lucide-react";
import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { HistoryEditor, OutreachHistoryEditor, ScoreStatusList } from "@/components/history-editor";
import { InlineNotice, Spinner } from "@/components/states";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Input, Label, Select } from "@/components/ui/primitives";
import { api, ApiError, unwrap, validationMessages, type Schemas } from "@/lib/api/client";
import { conditionLabel, fmtDate, humanize } from "@/lib/format";
import {
  emptyDraft,
  followUpPreview,
  historyCount,
  newSubmissionId,
  PROGRAMMES,
  registrationPayload,
  validateDraft,
  type Programme,
  type RegistrationDraft,
  SYNTHETIC_WARNING,
} from "@/lib/registration";

function FieldError({ text }: { text?: string }) {
  return text ? <p className="text-xs text-danger" role="alert">{text}</p> : null;
}

function RegisterPatient() {
  const qc = useQueryClient();
  const [draft, setDraft] = useState<RegistrationDraft>(emptyDraft);
  const [step, setStep] = useState<"edit" | "review">("edit");
  const [showErrors, setShowErrors] = useState(false);
  // One id per intended patient: retries (network errors, double clicks) reuse it, so the server
  // returns the already-created patient instead of creating a second one.
  const submissionId = useRef(newSubmissionId());
  const [result, setResult] = useState<Schemas["OnboardingResult"] | null>(null);
  // Synchronous guard: `isPending` only updates on the next render, so a fast double click could
  // otherwise send two requests (the server would still create one patient).
  const inFlight = useRef(false);

  const errors = useMemo(() => validateDraft(draft), [draft]);
  const programmes = PROGRAMMES.filter((p) => draft.conditions[p.id].enabled).map((p) => p.id);
  const visible = showErrors ? errors : {};
  const set = (patch: Partial<RegistrationDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const setCondition = (p: Programme, patch: Partial<RegistrationDraft["conditions"][Programme]>) =>
    setDraft((d) => ({ ...d, conditions: { ...d.conditions, [p]: { ...d.conditions[p], ...patch } } }));

  const mutation = useMutation({
    mutationFn: (allowDuplicate: boolean) =>
      unwrap(api.POST("/api/v1/patients", { body: registrationPayload(draft, submissionId.current, allowDuplicate) })),
    onSettled: () => {
      inFlight.current = false;
    },
    onSuccess: async (data) => {
      setResult(data);
      await qc.invalidateQueries();
    },
  });

  function toReview(e: React.FormEvent) {
    e.preventDefault();
    setShowErrors(true);
    if (Object.keys(errors).length === 0) {
      setStep("review");
      window.scrollTo({ top: 0 });
    }
  }

  function submit(allowDuplicate = false) {
    if (inFlight.current || result) return;
    inFlight.current = true;
    mutation.mutate(allowDuplicate);
  }

  function startAnother() {
    submissionId.current = newSubmissionId();
    setDraft(emptyDraft());
    setResult(null);
    setStep("edit");
    setShowErrors(false);
    mutation.reset();
  }

  const warning = (
    <div role="note" className="mb-5 flex items-start gap-2 rounded-lg border border-warn/30 bg-warn-soft px-4 py-3 text-[13px] font-medium text-warn">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      {SYNTHETIC_WARNING}
    </div>
  );

  if (result) {
    return (
      <>
        <PageHeader title="Patient registered" description={`${result.synthetic_mrn} · recorded ${fmtDate(result.recorded_at)}`} />
        {warning}
        <Card className="max-w-3xl">
          <CardHeader>
            <div>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="size-4 text-ok" />
                {result.created ? "The patient record was saved" : "This submission had already been saved"}
              </CardTitle>
              <CardDescription>
                {Object.entries(result.history_counts).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${humanize(k).toLowerCase()}`).join(" · ") || "No history entries"}
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="mb-2 text-[13px] font-semibold text-ink">Follow-up risk prediction</p>
              <ScoreStatusList items={result.score_status} />
            </div>
            {result.notes.map((n) => <InlineNotice key={n}>{n}</InlineNotice>)}
            <div className="flex flex-wrap gap-2">
              <Button asChild>
                <Link href={`/patients/${result.patient_id}`}>Open Patient 360</Link>
              </Button>
              <Button variant="outline" onClick={startAnother}>
                <UserPlus /> Register another patient
              </Button>
            </div>
          </CardContent>
        </Card>
      </>
    );
  }

  const apiError = mutation.error instanceof ApiError ? mutation.error : null;
  const duplicate = apiError?.code === "possible_duplicate" ? (apiError.details as { patient_id?: string; synthetic_mrn?: string }) : null;
  const fieldErrors = validationMessages(mutation.error);

  if (step === "review") {
    const payload = registrationPayload(draft, submissionId.current);
    return (
      <>
        <PageHeader title="Review registration" description="Check the details. Nothing is saved until you confirm." />
        {warning}
        <div className="grid max-w-5xl gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader><CardTitle>Patient</CardTitle></CardHeader>
            <CardContent className="space-y-1.5 text-[13px]">
              <p><span className="text-ink-subtle">Name:</span> {payload.given_name} {payload.family_name}</p>
              <p><span className="text-ink-subtle">Date of birth:</span> {fmtDate(payload.birth_date)} · {humanize(payload.sex)}</p>
              <p><span className="text-ink-subtle">Deprivation quintile:</span> {payload.deprivation_quintile} · language {payload.preferred_language}</p>
              <p><span className="text-ink-subtle">Registered with the practice:</span> {fmtDate(payload.registered_on)}</p>
              <p className="text-xs text-ink-subtle">The patient id and MRN are generated by the server.</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Expected next follow-up</CardTitle>
                <CardDescription>Derived from the latest review using the programme rules; the server confirms it</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-[13px]">
              {payload.conditions.map((c) => {
                const preview = followUpPreview(draft, c.programme);
                return (
                  <div key={c.programme} className="rounded-lg border border-border px-3 py-2">
                    <p className="font-medium">{conditionLabel(c.programme)} <span className="text-xs font-normal text-ink-subtle">since {fmtDate(c.onset_date)}</span></p>
                    {preview ? (
                      <p className="text-xs text-ink-muted">
                        Last review {fmtDate(preview.lastReview)} → due {fmtDate(preview.due)} · prediction time {fmtDate(preview.predictionAt)}
                      </p>
                    ) : (
                      <p className="text-xs text-warn">No completed review entered: no follow-up can be scheduled, so no prediction is possible yet.</p>
                    )}
                  </div>
                );
              })}
              <p className="text-xs text-ink-subtle">
                The model predicts at the prediction time (14 days before the due date) using only history recorded before it.
                History entered today is not visible to a prediction time that has already passed.
              </p>
            </CardContent>
          </Card>
          <Card className="lg:col-span-2">
            <CardHeader><CardTitle>History entries</CardTitle></CardHeader>
            <CardContent className="flex flex-wrap gap-2 text-[13px]">
              <Badge>{payload.history.encounters?.length ?? 0} visits</Badge>
              <Badge>{payload.history.appointments?.length ?? 0} appointments</Badge>
              <Badge>{payload.history.observations?.length ?? 0} observations</Badge>
              <Badge>{payload.history.medications?.length ?? 0} medications</Badge>
              <Badge>{payload.history.outreach?.length ?? 0} outreach attempts</Badge>
            </CardContent>
          </Card>
        </div>
        <div className="mt-5 max-w-5xl space-y-3">
          {duplicate && (
            <InlineNotice tone="warn">
              A registered patient with the same name and date of birth already exists ({duplicate.synthetic_mrn}).{" "}
              {duplicate.patient_id && <Link className="underline" href={`/patients/${duplicate.patient_id}`}>Open that record</Link>}{" "}
              or confirm this is a different person.
            </InlineNotice>
          )}
          {mutation.error && !duplicate && (
            <InlineNotice tone="danger">
              {apiError?.code === "network_error"
                ? "The API could not be reached. Submitting again is safe: it will not create a second patient."
                : apiError?.message ?? "Registration failed."}
              {fieldErrors.length > 0 && <ul className="mt-1 list-disc pl-4">{fieldErrors.map((f) => <li key={f}>{f}</li>)}</ul>}
            </InlineNotice>
          )}
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => setStep("edit")} disabled={mutation.isPending}>
              <Pencil /> Edit details
            </Button>
            {duplicate ? (
              <Button onClick={() => submit(true)} disabled={mutation.isPending}>
                {mutation.isPending ? <Spinner label="Saving…" /> : "Register as a different person"}
              </Button>
            ) : (
              <Button onClick={() => submit(false)} disabled={mutation.isPending}>
                {mutation.isPending ? <Spinner label="Saving…" /> : <><ClipboardCheck /> Confirm and register</>}
              </Button>
            )}
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Link href="/patients" className="mb-3 inline-flex items-center gap-1 text-[13px] text-ink-subtle hover:text-ink">
        <ArrowLeft className="size-3.5" /> Patients
      </Link>
      <PageHeader title="Register patient" description="Add a fictional patient with their chronic conditions and earlier care history." />
      {warning}
      <form onSubmit={toReview} className="max-w-5xl space-y-4" noValidate aria-label="Register patient">
        <Card>
          <CardHeader><CardTitle>Patient details</CardTitle></CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-1">
              <Label htmlFor="given">Given name</Label>
              <Input id="given" maxLength={80} autoComplete="off" value={draft.givenName} onChange={(e) => set({ givenName: e.target.value })} />
              <FieldError text={visible.givenName} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="family">Family name</Label>
              <Input id="family" maxLength={80} autoComplete="off" value={draft.familyName} onChange={(e) => set({ familyName: e.target.value })} />
              <FieldError text={visible.familyName} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="birth">Date of birth</Label>
              <Input id="birth" type="date" value={draft.birthDate} onChange={(e) => set({ birthDate: e.target.value })} />
              <FieldError text={visible.birthDate} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="sex">Sex</Label>
              <Select id="sex" value={draft.sex} onChange={(e) => set({ sex: e.target.value as RegistrationDraft["sex"] })}>
                <option value="">Choose…</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
              </Select>
              <FieldError text={visible.sex} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="imd">Deprivation quintile</Label>
              <Select id="imd" value={draft.deprivation} onChange={(e) => set({ deprivation: e.target.value })}>
                <option value="">Choose…</option>
                {[1, 2, 3, 4, 5].map((q) => (
                  <option key={q} value={q}>{q}{q === 1 ? " (least deprived)" : q === 5 ? " (most deprived)" : ""}</option>
                ))}
              </Select>
              <FieldError text={visible.deprivation} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="lang">Preferred language (code)</Label>
              <Input id="lang" maxLength={3} value={draft.language} onChange={(e) => set({ language: e.target.value.toLowerCase() })} />
              <FieldError text={visible.language} />
            </div>
            <div className="space-y-1">
              <Label htmlFor="reg">Registered with the practice on</Label>
              <Input id="reg" type="date" value={draft.registeredOn} onChange={(e) => set({ registeredOn: e.target.value })} />
              <FieldError text={visible.registeredOn} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle>Chronic conditions and programme</CardTitle>
              <CardDescription>Diabetes reviews are due every 182 days, hypertension reviews every 365 days</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {PROGRAMMES.map((p) => (
              <div key={p.id} className="flex flex-wrap items-end gap-3 rounded-lg border border-border p-3">
                <label className="flex min-w-48 items-center gap-2 text-[13px] font-medium text-ink">
                  <input type="checkbox" checked={draft.conditions[p.id].enabled} onChange={(e) => setCondition(p.id, { enabled: e.target.checked })} />
                  {p.label}
                </label>
                {draft.conditions[p.id].enabled && (
                  <div className="space-y-1">
                    <Label htmlFor={`onset-${p.id}`} className="text-xs">Diagnosed on</Label>
                    <Input id={`onset-${p.id}`} type="date" value={draft.conditions[p.id].onsetDate} onChange={(e) => setCondition(p.id, { onsetDate: e.target.value })} />
                  </div>
                )}
                <FieldError text={visible[`conditions.${p.id}`]} />
              </div>
            ))}
            <FieldError text={visible.conditions} />
          </CardContent>
        </Card>

        {programmes.length > 0 && (
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Earlier care history</CardTitle>
                <CardDescription>Optional, but a prediction needs at least one completed review. Times are in your local time zone.</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <HistoryEditor value={draft} onChange={(h) => set(h)} programmes={programmes} errors={visible} idPrefix="reg" />
              <OutreachHistoryEditor value={draft.outreach} onChange={(outreach) => set({ outreach })} programmes={programmes} errors={visible} />
            </CardContent>
          </Card>
        )}

        {showErrors && Object.keys(errors).length > 0 && (
          <InlineNotice tone="danger">Fix the {Object.keys(errors).length} highlighted problem(s) before reviewing.</InlineNotice>
        )}
        <div className="flex items-center gap-3">
          <Button type="submit"><ClipboardCheck /> Review {historyCount(draft) ? `(${historyCount(draft)} history entries)` : ""}</Button>
          <Button asChild variant="ghost"><Link href="/patients">Cancel</Link></Button>
        </div>
      </form>
    </>
  );
}

export default function NewPatientPage() {
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <RegisterPatient />
    </RoleGate>
  );
}
