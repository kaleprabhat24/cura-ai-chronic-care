"use client";

/* Patient 360 additions: the score-status panel for patients registered in Cura AI and the
 * "Add history" sheet (same validation and server logic as registration). */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, History, Sparkles } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { HistoryEditor, ScoreStatusList } from "@/components/history-editor";
import { OutreachForm } from "@/components/outreach-form";
import { InlineNotice, LoadingBlock, Spinner } from "@/components/states";
import { Sheet, SheetContent, SheetTrigger, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/overlay";
import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/primitives";
import { api, ApiError, unwrap, validationMessages, type Schemas } from "@/lib/api/client";
import { conditionLabel, fmtDate, fmtDateTime } from "@/lib/format";
import {
  emptyHistory,
  historyCount,
  historyPayload,
  newSubmissionId,
  validateHistory,
  type HistoryDraft,
  type Programme,
} from "@/lib/registration";

export function ScoreStatusPanel({ patientId }: { patientId: string }) {
  const qc = useQueryClient();
  const status = useQuery({
    queryKey: ["score-status", patientId],
    queryFn: () => unwrap(api.GET("/api/v1/patients/{patient_id}/score-status", { params: { path: { patient_id: patientId } } })),
  });
  const generate = useMutation({
    mutationFn: () => unwrap(api.POST("/api/v1/patients/{patient_id}/score", { params: { path: { patient_id: patientId } } })),
    onSuccess: async (data) => {
      qc.setQueryData(["score-status", patientId], data);
      await qc.invalidateQueries();
    },
  });
  const items = status.data?.score_status ?? [];
  const ready = items.some((i) => i.status === "ready_to_score");
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle>Prediction status</CardTitle>
          <CardDescription>
            Registered in Cura AI · scored with the existing model at each follow-up&apos;s prediction time
            {status.data ? ` · checked ${fmtDateTime(status.data.evaluated_at)}` : ""}
          </CardDescription>
        </div>
        {ready && (
          <Button size="sm" onClick={() => generate.mutate()} disabled={generate.isPending}>
            {generate.isPending ? <Spinner label="Scoring…" /> : <><Sparkles /> Generate score</>}
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-2">
        {status.isLoading ? <LoadingBlock rows={2} /> : status.error ? (
          <InlineNotice tone="danger">{status.error instanceof ApiError ? status.error.message : "Could not load the prediction status."}</InlineNotice>
        ) : (
          <ScoreStatusList items={items} />
        )}
        {generate.error && (
          <InlineNotice tone="danger">{generate.error instanceof ApiError ? generate.error.message : "Scoring failed."}</InlineNotice>
        )}
      </CardContent>
    </Card>
  );
}

function HistoryForm({ patientId, programmes, birthDate, onSaved }: {
  patientId: string;
  programmes: Programme[];
  birthDate: string;
  onSaved: (r: Schemas["OnboardingResult"]) => void;
}) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState<HistoryDraft>(emptyHistory);
  const [showErrors, setShowErrors] = useState(false);
  const submissionId = useRef(newSubmissionId());
  const inFlight = useRef(false);
  const errors = useMemo(() => validateHistory(draft, new Date(), new Set(programmes), "", birthDate), [draft, programmes, birthDate]);
  const mutation = useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/v1/patients/{patient_id}/history", {
        params: { path: { patient_id: patientId } },
        body: { submission_id: submissionId.current, history: historyPayload(draft) },
      })),
    onSettled: () => {
      inFlight.current = false;
    },
    onSuccess: async (data) => {
      submissionId.current = newSubmissionId();
      setDraft(emptyHistory());
      setShowErrors(false);
      await qc.invalidateQueries();
      onSaved(data);
    },
  });

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setShowErrors(true);
    if (inFlight.current || historyCount(draft) === 0 || Object.keys(errors).length) return;
    inFlight.current = true;
    mutation.mutate();
  }

  const fieldErrors = validationMessages(mutation.error);
  return (
    <form onSubmit={submit} className="space-y-3" noValidate aria-label="Add history">
      <HistoryEditor value={draft} onChange={setDraft} programmes={programmes} errors={showErrors ? errors : {}} idPrefix="add" />
      {showErrors && historyCount(draft) === 0 && <InlineNotice tone="warn">Add at least one entry.</InlineNotice>}
      {mutation.error && (
        <InlineNotice tone="danger">
          {mutation.error instanceof ApiError && mutation.error.code === "network_error"
            ? "The API could not be reached. Saving again is safe: entries are not duplicated."
            : mutation.error instanceof ApiError ? mutation.error.message : "Saving failed."}
          {fieldErrors.length > 0 && <ul className="mt-1 list-disc pl-4">{fieldErrors.map((f) => <li key={f}>{f}</li>)}</ul>}
        </InlineNotice>
      )}
      <Button type="submit" disabled={mutation.isPending}>
        {mutation.isPending ? <Spinner label="Saving…" /> : `Save ${historyCount(draft) || ""} entries`}
      </Button>
    </form>
  );
}

export function AddHistorySheet({ patientId, programmes, birthDate, outreachEpisode }: {
  patientId: string;
  programmes: Programme[];
  birthDate: string;
  outreachEpisode?: { id: string; programme: string; due_date: string };
}) {
  const [open, setOpen] = useState(false);
  const [saved, setSaved] = useState<Schemas["OnboardingResult"] | null>(null);
  return (
    <Sheet open={open} onOpenChange={(o) => { setOpen(o); if (!o) setSaved(null); }}>
      <SheetTrigger asChild>
        <Button variant="outline"><History /> Add history</Button>
      </SheetTrigger>
      <SheetContent title="Add history" className="max-w-2xl"
                    description="Entries are stored with today's recording time; they never change predictions made earlier.">
        <Tabs defaultValue="clinical">
          <TabsList className="mb-4">
            <TabsTrigger value="clinical">Clinical history</TabsTrigger>
            <TabsTrigger value="outreach">Outreach</TabsTrigger>
          </TabsList>
          <TabsContent value="clinical" className="space-y-3">
            {saved && (
              <InlineNotice tone="ok">
                <span className="inline-flex items-center gap-1.5 font-medium"><CheckCircle2 className="size-3.5" /> Saved
                  {" "}{Object.entries(saved.history_counts).filter(([, n]) => n > 0).map(([k, n]) => `${n} ${k}`).join(", ")}.</span>
                {saved.score_status.length > 0 && <div className="mt-2"><ScoreStatusList items={saved.score_status} /></div>}
                {saved.notes.map((n) => <p key={n} className="mt-1">{n}</p>)}
              </InlineNotice>
            )}
            <HistoryForm patientId={patientId} programmes={programmes} birthDate={birthDate} onSaved={setSaved} />
          </TabsContent>
          <TabsContent value="outreach">
            {outreachEpisode ? (
              <>
                <p className="mb-3 text-[13px] text-ink-muted">
                  For the {conditionLabel(outreachEpisode.programme)} follow-up due {fmtDate(outreachEpisode.due_date)}.
                </p>
                <OutreachForm episodeId={outreachEpisode.id} />
              </>
            ) : (
              <InlineNotice tone="warn">
                Outreach is recorded against an eligible follow-up episode, and this patient has none yet. Add a review visit first.
              </InlineNotice>
            )}
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}
