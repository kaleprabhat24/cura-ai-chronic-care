"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, FileJson, PhoneCall } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { COLORS, SimpleLines } from "@/components/charts";
import { FhirExportButton } from "@/components/fhir-export";
import { OutreachForm } from "@/components/outreach-form";
import { AddHistorySheet, ScoreStatusPanel } from "@/components/patient-onboarding";
import {
  ExplanationFactors,
  FollowUpBadge,
  ProvenanceTag,
  ScoreContextNotice,
  ScoreProvenance,
  TierBadge,
} from "@/components/score";
import { EmptyState, ErrorState, InlineNotice, LoadingBlock } from "@/components/states";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/overlay";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/primitives";
import { api, unwrap, type Schemas } from "@/lib/api/client";
import { conditionLabel, fmtDate, fmtDateTime, humanize, OUTCOME_LABELS, pct } from "@/lib/format";
import { FOLLOW_UP_LABELS, presentScore } from "@/lib/score";
import type { Programme } from "@/lib/registration";
import { cn } from "@/lib/utils";

const HBA1C = "4548-4";
const SBP = "8480-6";
const DBP = "8462-4";
const PROGRAMME_BY_CODE: Record<string, Programme> = { "44054006": "t2d", "38341003": "htn" };
const EVENT_FILTERS = [
  { id: "all", label: "All" },
  { id: "encounter", label: "Visits" },
  { id: "appointment", label: "Appointments" },
  { id: "observation", label: "Observations" },
  { id: "medication", label: "Medications" },
  { id: "outreach", label: "Outreach" },
  { id: "risk_score", label: "Scores" },
] as const;

function Patient360() {
  const { id } = useParams<{ id: string }>();
  const params = useSearchParams();
  const router = useRouter();
  const [sheetOpen, setSheetOpen] = useState(false);
  const [eventFilter, setEventFilter] = useState<(typeof EVENT_FILTERS)[number]["id"]>("all");

  const detail = useQuery({
    queryKey: ["patient", id],
    queryFn: () => unwrap(api.GET("/api/v1/patients/{patient_id}", { params: { path: { patient_id: id } } })),
  });
  const timeline = useQuery({
    queryKey: ["timeline", id],
    queryFn: () =>
      unwrap(api.GET("/api/v1/patients/{patient_id}/timeline", { params: { path: { patient_id: id }, query: { limit: 1000 } } })),
  });

  const episodes = useMemo(() => detail.data?.episodes ?? [], [detail.data]);
  const selectedId =
    params.get("episode") ??
    episodes.find((e) => e.eligible && e.latest_prediction && ["upcoming", "due_in_grace_window", "overdue_unresolved"].includes(e.score_context.follow_up_status))?.id ??
    episodes.find((e) => e.eligible)?.id ??
    episodes[0]?.id;
  const episode = episodes.find((e) => e.id === selectedId);
  const scoreId = episode?.latest_prediction?.risk_score_id;

  const score = useQuery({
    queryKey: ["risk-score", scoreId],
    enabled: !!scoreId,
    queryFn: () => unwrap(api.GET("/api/v1/risk-scores/{risk_score_id}", { params: { path: { risk_score_id: scoreId! } } })),
  });
  const history = useQuery({
    queryKey: ["score-history", selectedId],
    enabled: !!selectedId,
    queryFn: () =>
      unwrap(api.GET("/api/v1/episodes/{episode_id}/risk-scores", { params: { path: { episode_id: selectedId! } } })),
  });
  const outreach = useQuery({
    queryKey: ["episode-outreach", selectedId],
    enabled: !!selectedId,
    queryFn: () =>
      unwrap(api.GET("/api/v1/episodes/{episode_id}/outreach-attempts", { params: { path: { episode_id: selectedId! } } })),
  });

  const events = useMemo(() => timeline.data?.items ?? [], [timeline.data]);
  const trends = useMemo(() => {
    const obs = events.filter((e) => e.type === "observation").slice().reverse();
    const pick = (code: string) => obs.filter((e) => e.detail.code === code);
    const hba1c = pick(HBA1C).map((e) => ({ date: fmtDate(e.occurred_at), value: e.detail.value as number }));
    const bpByDate = new Map<string, { date: string; sbp?: number; dbp?: number }>();
    for (const e of [...pick(SBP), ...pick(DBP)]) {
      const key = e.occurred_at.slice(0, 16);
      const row = bpByDate.get(key) ?? { date: fmtDate(e.occurred_at) };
      if (e.detail.code === SBP) row.sbp = e.detail.value as number;
      else row.dbp = e.detail.value as number;
      bpByDate.set(key, row);
    }
    const bp = [...bpByDate.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, v]) => v);
    const scores = events
      .filter((e) => e.type === "risk_score")
      .slice()
      .reverse()
      .map((e) => ({ date: fmtDate(e.occurred_at), probability: e.detail.probability as number }));
    return { hba1c, bp, scores };
  }, [events]);

  if (detail.isLoading) return <LoadingBlock rows={10} />;
  if (detail.error) return <ErrorState error={detail.error} onRetry={() => detail.refetch()} />;
  const d = detail.data!;
  const p = d.patient;
  const filtered = events.filter((e) =>
    eventFilter === "all" ? true : eventFilter === "medication" ? e.type.startsWith("medication") : e.type === eventFilter,
  );
  const view = episode ? presentScore(episode.score_context) : null;
  const isApp = p.id.startsWith("pt_app_");
  const programmes = [...new Set(d.conditions.map((c) => PROGRAMME_BY_CODE[c.code]).filter(Boolean))];
  const outreachEpisode = episode?.eligible ? episode : episodes.find((e) => e.eligible);

  return (
    <>
      <Link href={isApp ? "/patients" : "/worklist"} className="mb-3 inline-flex items-center gap-1 text-[13px] text-ink-subtle hover:text-ink">
        <ArrowLeft className="size-3.5" /> {isApp ? "Patients" : "Worklist"}
      </Link>
      <PageHeader
        title={`${p.given_name} ${p.family_name}`}
        description={`${p.synthetic_mrn} · ${p.age_years} years · ${p.sex} · deprivation quintile ${p.deprivation_quintile} · language ${p.preferred_language} · registered ${fmtDate(p.registered_on)}`}
        actions={
          <div className="flex flex-wrap gap-2">
            <FhirExportButton patientId={p.id} />
            <AddHistorySheet patientId={p.id} programmes={programmes} birthDate={p.birth_date}
                             outreachEpisode={outreachEpisode} />
            {episode?.eligible && (
              <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
                <SheetTrigger asChild>
                  <Button>
                    <PhoneCall /> Record outreach
                  </Button>
                </SheetTrigger>
                <SheetContent title="Record outreach" description={`${conditionLabel(episode.programme)} review due ${fmtDate(episode.due_date)}`}>
                  <OutreachForm episodeId={episode.id} />
                </SheetContent>
              </Sheet>
            )}
          </div>
        }
      />
      <div className="mb-4 flex flex-wrap gap-2">
        <ProvenanceTag provenance={p.provenance} />
        {isApp && <Badge tone="info">Registered in Cura AI</Badge>}
        {p.death_date && <Badge tone="danger">Died {fmtDate(p.death_date)}</Badge>}
        {p.transfer_out_date && <Badge tone="warn">Transferred out {fmtDate(p.transfer_out_date)}</Badge>}
        {d.conditions.map((c) => (
          <Badge key={c.id} tone="brand">
            {c.display} · since {fmtDate(c.onset_date)}
          </Badge>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          {isApp && <ScoreStatusPanel patientId={p.id} />}
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Follow-up episode</CardTitle>
                <CardDescription>Select an episode to see its score, explanation and outreach</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {episodes.filter((e) => e.eligible || isApp).slice(0, 12).map((e) => (
                  <button
                    key={e.id}
                    onClick={() => router.replace(`/patients/${p.id}?episode=${e.id}`, { scroll: false })}
                    className={cn(
                      "rounded-lg border px-3 py-1.5 text-left text-xs transition-colors",
                      e.id === selectedId ? "border-brand bg-brand-soft text-brand-strong" : "border-border hover:bg-surface-muted",
                    )}
                  >
                    <span className="font-medium">{conditionLabel(e.programme)}</span> · due {fmtDate(e.due_date)}
                  </button>
                ))}
              </div>
              {!episode ? (
                <EmptyState title="No follow-up episodes" />
              ) : (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <FollowUpBadge ctx={episode.score_context} />
                    <Badge>Due {fmtDate(episode.due_date)}</Badge>
                    <Badge>Outcome window ends {fmtDate(episode.window_end_date)}</Badge>
                    <ProvenanceTag provenance={episode.recorded_outcome.provenance} />
                    <span className="text-xs text-ink-subtle">
                      Recorded status: {FOLLOW_UP_LABELS[episode.score_context.follow_up_status]} · label{" "}
                      {humanize(episode.recorded_outcome.label_status)}
                    </span>
                  </div>
                  <ScoreContextNotice ctx={episode.score_context} />
                  {episode.latest_prediction ? (
                    <div className="grid gap-4 md:grid-cols-[220px_minmax(0,1fr)]">
                      <div className="rounded-lg border border-border p-4">
                        <ProvenanceTag provenance="simulated_prediction" />
                        <p className={cn("tabular mt-3 text-[32px] font-semibold leading-none",
                                         view?.showProbabilityAsCurrent ? "text-ink" : "text-ink-subtle")}>
                          {pct(episode.latest_prediction.probability)}
                        </p>
                        <p className="mt-1 text-xs text-ink-subtle">
                          {view?.showProbabilityAsCurrent
                            ? "Probability the follow-up is still incomplete 30 days after the due date"
                            : "Historical probability — not a current prediction"}
                        </p>
                        <div className="mt-3">
                          <TierBadge tier={episode.latest_prediction.risk_tier} muted={!view?.showProbabilityAsCurrent} />
                        </div>
                        <div className="mt-4">
                          <ScoreProvenance prediction={episode.latest_prediction} />
                        </div>
                      </div>
                      <div>
                        <p className="mb-2 text-[13px] font-semibold text-ink">Why the model scored it this way</p>
                        {score.isLoading ? (
                          <LoadingBlock rows={5} />
                        ) : score.error ? (
                          <ErrorState error={score.error} />
                        ) : score.data ? (
                          <>
                            <ExplanationFactors factors={score.data.explanation.factors} />
                            <p className="mt-3 text-[11px] leading-relaxed text-ink-subtle">
                              {score.data.explanation.units} {score.data.explanation.note} Method:{" "}
                              {humanize(score.data.explanation.method)}; base log-odds {score.data.explanation.base_log_odds.toFixed(2)}{" "}
                              → model log-odds {score.data.explanation.model_log_odds.toFixed(2)}.
                            </p>
                          </>
                        ) : null}
                      </div>
                    </div>
                  ) : (
                    <InlineNotice tone="warn">
                      {isApp
                        ? "No score for this episode yet. The prediction status above explains why."
                        : "No usable score for this episode in the latest scoring run"}
                      {!isApp && (episode.score_context.data_sufficiency === "limited_history" ? " (limited history)." : ".")}
                    </InlineNotice>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle>HbA1c</CardTitle>
                  <CardDescription>Recorded results (%)</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                {trends.hba1c.length ? (
                  <SimpleLines data={trends.hba1c} x="date" lines={[{ key: "value", label: "HbA1c %", color: COLORS.brand }]}
                               height={190} reference={{ y: 7, label: "7%" }} />
                ) : (
                  <EmptyState title="No HbA1c recorded" hint="Missing results are shown as gaps, never filled in." />
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div>
                  <CardTitle>Blood pressure</CardTitle>
                  <CardDescription>Recorded readings (mm[Hg])</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                {trends.bp.length ? (
                  <SimpleLines data={trends.bp} x="date" height={190}
                               lines={[{ key: "sbp", label: "Systolic", color: COLORS.high }, { key: "dbp", label: "Diastolic", color: COLORS.info }]} />
                ) : (
                  <EmptyState title="No blood pressure recorded" />
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Longitudinal timeline</CardTitle>
                <CardDescription>
                  {timeline.data ? `${timeline.data.total} events — recorded history, outreach and simulated scores (labelled)` : "Loading"}
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="mb-3 flex flex-wrap gap-1.5">
                {EVENT_FILTERS.map((f) => (
                  <button key={f.id} onClick={() => setEventFilter(f.id)}
                          className={cn("rounded-md border px-2.5 py-1 text-xs", eventFilter === f.id ? "border-brand bg-brand-soft text-brand-strong" : "border-border text-ink-muted hover:bg-surface-muted")}>
                    {f.label}
                  </button>
                ))}
              </div>
              {timeline.isLoading ? (
                <LoadingBlock rows={6} />
              ) : timeline.error ? (
                <ErrorState error={timeline.error} />
              ) : filtered.length === 0 ? (
                <EmptyState title="No events of this type" />
              ) : (
                <ol className="max-h-[560px] space-y-0 overflow-y-auto pr-1">
                  {filtered.slice(0, 250).map((e) => (
                    <li key={`${e.type}-${e.id}`} className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 border-l-2 border-border py-2 pl-3 text-[13px]">
                      <span className="tabular text-xs text-ink-subtle">{fmtDate(e.occurred_at)}</span>
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium text-ink">{e.title}</span>
                          <ProvenanceTag provenance={e.provenance} />
                        </div>
                        {e.recorded_at && e.recorded_at !== e.occurred_at && (
                          <p className="text-[11px] text-ink-subtle">Recorded {fmtDateTime(e.recorded_at)}</p>
                        )}
                        {e.type === "outreach" && e.detail.note ? (
                          <p className="text-xs text-ink-muted">{String(e.detail.note)}</p>
                        ) : null}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Score history</CardTitle>
                <CardDescription>All scoring snapshots — none are deleted</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {trends.scores.length > 1 && (
                <SimpleLines data={trends.scores} x="date" height={150} yDomain={[0, 1]} yFormatter={(v) => pct(v)}
                             lines={[{ key: "probability", label: "Probability", color: COLORS.brand }]} />
              )}
              {history.data && history.data.length > 0 ? (
                <ul className="mt-2 divide-y divide-border text-[13px]">
                  {history.data.map((h) => (
                    <li key={h.id} className="flex items-center justify-between py-2">
                      <span>
                        Run #{h.scoring_run_id} · {h.model_version}
                        <span className="block text-[11px] text-ink-subtle">
                          predicted {fmtDate(h.prediction_at)} · scored {fmtDateTime(h.scored_at)}
                        </span>
                      </span>
                      <span className="flex items-center gap-2">
                        <span className="tabular">{pct(h.probability)}</span>
                        <TierBadge tier={h.risk_tier} muted={!view?.showProbabilityAsCurrent} />
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-[13px] text-ink-subtle">No scores for this episode.</p>
              )}
              {score.data && (
                <p className="mt-3 text-[11px] text-ink-subtle">
                  Latest run #{score.data.run.id}: {score.data.run.model_type.replace("_", " ")}, {score.data.run.calibration_method} calibration,
                  feature spec {score.data.run.feature_spec_version}, data {score.data.run.data_fingerprint.slice(0, 12)}…
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Outreach for this episode</CardTitle>
                <CardDescription>Recorded attempts, newest first</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {outreach.isLoading ? (
                <LoadingBlock rows={3} />
              ) : outreach.error ? (
                <ErrorState error={outreach.error} />
              ) : !outreach.data?.length ? (
                <EmptyState title="No outreach recorded" />
              ) : (
                <ul className="space-y-2.5">
                  {outreach.data.slice(0, 15).map((o: Schemas["OutreachOut"]) => (
                    <li key={o.id} className="rounded-lg border border-border px-3 py-2 text-[13px]">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{OUTCOME_LABELS[o.outcome] ?? o.outcome}</span>
                        <Badge tone={o.origin === "app" ? "ok" : "neutral"}>{o.origin === "app" ? "Cura AI" : "Historical"}</Badge>
                      </div>
                      <p className="text-xs text-ink-subtle">
                        {humanize(o.channel)} · {fmtDateTime(o.attempted_at)} · {o.recorded_by}
                      </p>
                      {o.note && <p className="mt-1 text-xs text-ink-muted">{o.note}</p>}
                      {o.next_action !== "none" && (
                        <p className="mt-1 text-xs text-warn">
                          Next: {humanize(o.next_action)}
                          {o.next_action_due ? ` by ${fmtDate(o.next_action_due)}` : ""}
                        </p>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Active medicines</CardTitle>
                <CardDescription>Recorded medication statements</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {d.active_medications.length === 0 ? (
                <p className="text-[13px] text-ink-subtle">None recorded.</p>
              ) : (
                <ul className="space-y-1.5 text-[13px]">
                  {d.active_medications.map((mm) => (
                    <li key={mm.id} className="flex justify-between">
                      <span className="capitalize">{mm.display}</span>
                      <span className="text-xs text-ink-subtle">since {fmtDate(mm.start_date)}</span>
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-4 space-y-1.5 text-[13px]">
                {d.latest_observations.map((o) => (
                  <div key={o.id} className="flex justify-between">
                    <span className="text-ink-subtle">{o.display}</span>
                    <span className="tabular">
                      {o.value} {o.unit} <span className="text-[11px] text-ink-subtle">({fmtDate(o.effective_at)})</span>
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <p className="flex items-center gap-1.5 text-[11px] text-ink-subtle">
            <FileJson className="size-3.5" /> Synthetic record. Export available as a FHIR R4-oriented Bundle.
          </p>
        </div>
      </div>
    </>
  );
}

export default function PatientPage() {
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <Suspense fallback={<LoadingBlock rows={10} />}>
        <Patient360 />
      </Suspense>
    </RoleGate>
  );
}
