"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2 } from "lucide-react";
import { useState } from "react";
import { api, ApiError, unwrap, validationMessages, type Schemas } from "@/lib/api/client";
import { CHANNELS, NEXT_ACTIONS, OUTCOMES, OUTCOME_LABELS, humanize } from "@/lib/format";
import { Button, Input, Label, Select, Textarea } from "@/components/ui/primitives";
import { InlineNotice, Spinner } from "@/components/states";

type Body = Schemas["OutreachCreate"];

function localNowValue(): string {
  const d = new Date();
  d.setSeconds(0, 0);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

/** Records an outreach attempt. Success is only shown after the API has confirmed the write (201). */
export function OutreachForm({ episodeId, onSaved }: { episodeId: string; onSaved?: (o: Schemas["OutreachOut"]) => void }) {
  const qc = useQueryClient();
  const [channel, setChannel] = useState<Body["channel"]>("phone");
  const [outcome, setOutcome] = useState<Body["outcome"] | "">("");
  const [attemptedAt, setAttemptedAt] = useState(localNowValue());
  const [note, setNote] = useState("");
  const [nextAction, setNextAction] = useState<NonNullable<Body["next_action"]>>("none");
  const [nextDue, setNextDue] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const [saved, setSaved] = useState<Schemas["OutreachOut"] | null>(null);

  const mutation = useMutation({
    mutationFn: (body: Body) =>
      unwrap(
        api.POST("/api/v1/episodes/{episode_id}/outreach-attempts", {
          params: { path: { episode_id: episodeId } },
          body,
        }),
      ),
    onSuccess: async (created) => {
      setSaved(created);
      setNote("");
      setOutcome("");
      setNextAction("none");
      setNextDue("");
      await qc.invalidateQueries();
      onSaved?.(created);
    },
  });

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setSaved(null);
    setLocalError(null);
    if (!outcome) return setLocalError("Choose the outcome of the contact attempt.");
    if (nextAction !== "none" && !nextDue) return setLocalError("Set a due date for the next action.");
    const when = new Date(attemptedAt);
    if (Number.isNaN(when.getTime())) return setLocalError("Enter a valid attempt time.");
    mutation.mutate({
      channel,
      outcome,
      attempted_at: when.toISOString(),
      note: note.trim() || null,
      next_action: nextAction,
      next_action_due: nextAction === "none" ? null : nextDue,
    });
  }

  const apiError = mutation.error;
  const fieldErrors = validationMessages(apiError);
  return (
    <form onSubmit={submit} className="space-y-4" aria-label="Record outreach">
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="channel">Channel</Label>
          <Select id="channel" value={channel} onChange={(e) => setChannel(e.target.value as Body["channel"])}>
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {humanize(c)}
              </option>
            ))}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="attempted">Attempted at</Label>
          <Input id="attempted" type="datetime-local" value={attemptedAt} max={localNowValue()}
                 onChange={(e) => setAttemptedAt(e.target.value)} />
        </div>
      </div>
      <fieldset className="space-y-1.5">
        <legend className="text-[13px] font-medium text-ink">Outcome</legend>
        <div className="grid grid-cols-2 gap-2">
          {OUTCOMES.map((o) => (
            <label
              key={o}
              className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-[13px] transition-colors ${
                outcome === o ? "border-brand bg-brand-soft text-brand-strong" : "border-border hover:bg-surface-muted"
              }`}
            >
              <input type="radio" name="outcome" value={o} checked={outcome === o}
                     onChange={() => setOutcome(o)} className="accent-[var(--color-brand)]" />
              {OUTCOME_LABELS[o]}
            </label>
          ))}
        </div>
      </fieldset>
      <div className="space-y-1.5">
        <Label htmlFor="note">Note (optional)</Label>
        <Textarea id="note" maxLength={2000} value={note} onChange={(e) => setNote(e.target.value)}
                  placeholder="What happened? Avoid unnecessary personal detail." />
        <p className="text-right text-[11px] text-ink-subtle">{note.length}/2000</p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="next">Next action</Label>
          <Select id="next" value={nextAction}
                  onChange={(e) => setNextAction(e.target.value as NonNullable<Body["next_action"]>)}>
            {NEXT_ACTIONS.map((a) => (
              <option key={a} value={a}>
                {humanize(a)}
              </option>
            ))}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="nextDue">Next action due</Label>
          <Input id="nextDue" type="date" value={nextDue} disabled={nextAction === "none"}
                 onChange={(e) => setNextDue(e.target.value)} />
        </div>
      </div>
      {localError && <InlineNotice tone="danger">{localError}</InlineNotice>}
      {apiError && (
        <InlineNotice tone="danger">
          <p className="font-medium">
            {apiError instanceof ApiError ? apiError.message : "The outreach could not be saved."}
          </p>
          {fieldErrors.length > 0 && (
            <ul className="mt-1 list-disc pl-4">
              {fieldErrors.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          )}
        </InlineNotice>
      )}
      {saved && !mutation.isPending && (
        <InlineNotice tone="ok">
          <span className="inline-flex items-center gap-1.5">
            <CheckCircle2 className="size-4" /> Saved: {OUTCOME_LABELS[saved.outcome] ?? saved.outcome} recorded by{" "}
            {saved.recorded_by}.
          </span>
        </InlineNotice>
      )}
      <Button type="submit" className="w-full" disabled={mutation.isPending}>
        {mutation.isPending ? <Spinner label="Saving…" /> : "Record outreach"}
      </Button>
    </form>
  );
}
