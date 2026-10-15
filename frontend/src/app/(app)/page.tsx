"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CircleCheck, CircleAlert } from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/app-shell";
import { COLORS, SimpleBars } from "@/components/charts";
import { Stat } from "@/components/stat";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/states";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { hasRole, useAuth } from "@/lib/auth/auth-context";
import { fmtDate, fmtDateTime, num, OUTCOME_LABELS, pct } from "@/lib/format";

export default function DashboardPage() {
  const { me } = useAuth();
  const clinical = hasRole(me, "coordinator", "admin");
  const analytics = hasRole(me, "analyst", "admin");
  const summary = useQuery({ queryKey: ["dashboard"], queryFn: () => unwrap(api.GET("/api/v1/dashboard/summary")) });
  const outcomes = useQuery({ queryKey: ["outcomes"], queryFn: () => unwrap(api.GET("/api/v1/outreach/outcomes")) });
  const health = useQuery({ queryKey: ["health"], queryFn: () => unwrap(api.GET("/api/v1/health")) });
  const recent = useQuery({
    queryKey: ["outreach-recent"],
    enabled: clinical,
    queryFn: () => unwrap(api.GET("/api/v1/outreach-attempts", { params: { query: { page_size: 6 } } })),
  });
  const dq = useQuery({
    queryKey: ["data-quality"],
    enabled: analytics,
    queryFn: () => unwrap(api.GET("/api/v1/data-quality")),
  });

  if (summary.isLoading) return <LoadingBlock rows={8} />;
  if (summary.error) return <ErrorState error={summary.error} onRetry={() => summary.refetch()} />;
  const d = summary.data!;
  const run = d.latest_scoring_run;
  const predictiveHigh = d.predictive_by_tier?.high ?? 0;
  const statusData = [
    { status: "Upcoming", n: d.open_upcoming ?? 0 },
    { status: "Due — window open", n: d.open_in_grace_window ?? 0 },
    { status: "Overdue — unresolved", n: d.open_overdue_unresolved ?? 0 },
  ];
  const tierData = (["high", "medium", "low"] as const).map((t) => ({
    tier: t[0].toUpperCase() + t.slice(1),
    n: d.predictive_by_tier?.[t] ?? 0,
  }));
  const weekData = d.due_by_week.map((w) => ({ week: fmtDate(w.week_start).slice(0, 6), n: w.due_open_episodes }));
  const comp = d.completion_last_12_months as { completed_on_time_rate?: number | null; labelled_episodes?: number };
  const hist = outcomes.data?.attempts_by_outcome?.historical_simulation ?? {};
  const reached = (hist.reached_booked ?? 0) + (hist.reached_declined ?? 0) + (hist.reached_already_done ?? 0);

  return (
    <>
      <PageHeader
        title="Operations dashboard"
        description={`Synthetic cohort as of ${fmtDate(d.as_of)} (the dataset cutoff). Risk tiers are simulated predictions from scoring run #${run?.id ?? "—"}.`}
        actions={
          clinical && (
            <Button asChild>
              <Link href="/worklist">
                Open worklist <ArrowRight />
              </Link>
            </Button>
          )
        }
      />
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Open eligible follow-ups" value={num(d.open_episodes)} hint={`${num(d.patients_active)} active patients of ${num(d.patients_total)}`} />
        <Stat label="High priority (predictive)" value={num(predictiveHigh)} tone="brand"
              hint={`${num(Object.values(d.predictive_by_tier ?? {}).reduce((a, b) => a + b, 0))} with a live prediction`} />
        <Stat label="Overdue — unresolved" value={num(d.open_overdue_unresolved)} tone="danger"
              hint="Past the 30-day window: recorded as not completed" />
        <Stat label="Due in next 30 days" value={num(d.due_next_30_days)} hint={`${num(d.due_next_14_days)} within 14 days`} />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div>
              <CardTitle>Open follow-ups by status</CardTitle>
              <CardDescription>
                Overdue cases need operational attention; their stored scores are historical, not current predictions.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={statusData} x="status" bars={[{ key: "n", label: "Episodes", color: COLORS.brand }]}
                        colors={[COLORS.brandSoft, COLORS.medium, COLORS.high]} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Priority distribution</CardTitle>
              <CardDescription>Open follow-ups with a predictive score</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={tierData} x="tier" bars={[{ key: "n", label: "Episodes", color: COLORS.brand }]}
                        colors={[COLORS.high, COLORS.medium, COLORS.low]} />
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Due-date load</CardTitle>
              <CardDescription>Open follow-ups due in the 8 weeks after the cutoff</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={weekData} x="week" bars={[{ key: "n", label: "Due", color: COLORS.brand }]} height={200} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Outreach</CardTitle>
              <CardDescription>Recorded attempts (historical simulation + entered in the app)</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-[13px]">
            <Row label="Historical attempts" value={num(d.outreach_historical_total)} />
            <Row label="Entered in Cura AI" value={num(d.outreach_app_total)} />
            <Row label="Last 30 days before cutoff" value={num(d.outreach_last_30_days)} />
            <Row label="Phone contact rate (historical)" value={pct(outcomes.data?.phone_contact_rate?.historical_simulation ?? null)} />
            <Row label="Successful contacts (historical)" value={num(reached)} />
            <Row label="Completed on time (last 12 mo)"
                 value={`${pct(comp.completed_on_time_rate ?? null)} of ${num(comp.labelled_episodes ?? 0)}`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Model & data status</CardTitle>
              <CardDescription>Synthetic-data evaluation, not clinical validation</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-[13px]">
            <Row label="Model" value={run ? `${run.model_version} (${run.model_type.replace("_", " ")})` : "—"} />
            <Row label="Scored" value={run ? `${num(run.n_scored)} episodes · ${fmtDateTime(run.finished_at)}` : "No scoring run"} />
            <Row label="API health" value={health.data ? <StatusBadge ok={health.data.status === "ok"} text={health.data.status} /> : "…"} />
            {analytics && dq.data && (
              <Row label="Data-quality checks"
                   value={<StatusBadge ok={dq.data.status === "ok"} text={`${dq.data.checks.filter((c) => c.ok).length}/${dq.data.checks.length} passing`} />} />
            )}
            <Link href="/model" className="inline-flex items-center gap-1 text-[13px] font-medium text-brand hover:underline">
              Model & data health <ArrowRight className="size-3.5" />
            </Link>
          </CardContent>
        </Card>
      </div>

      {clinical && (
        <Card className="mt-4">
          <CardHeader>
            <div>
              <CardTitle>Recent outreach activity</CardTitle>
              <CardDescription>Latest recorded attempts</CardDescription>
            </div>
            <Button asChild variant="outline" size="sm">
              <Link href="/outreach">All outreach</Link>
            </Button>
          </CardHeader>
          <CardContent>
            {recent.isLoading ? (
              <LoadingBlock rows={3} />
            ) : recent.error ? (
              <ErrorState error={recent.error} />
            ) : recent.data!.items.length === 0 ? (
              <EmptyState title="No outreach recorded yet" />
            ) : (
              <ul className="divide-y divide-border">
                {recent.data!.items.map((o) => (
                  <li key={o.id} className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-[13px]">
                    <Link href={`/patients/${o.patient_id}`} className="font-medium text-ink hover:text-brand">
                      {o.patient.display_name} <span className="text-ink-subtle">· {o.patient.synthetic_mrn}</span>
                    </Link>
                    <span className="flex items-center gap-2 text-ink-muted">
                      <Badge tone={o.origin === "app" ? "ok" : "neutral"}>{o.origin === "app" ? "Cura AI" : "Historical"}</Badge>
                      {OUTCOME_LABELS[o.outcome] ?? o.outcome} · {o.channel} · {fmtDateTime(o.recorded_at)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      )}
    </>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-ink-subtle">{label}</span>
      <span className="tabular text-right font-medium text-ink">{value}</span>
    </div>
  );
}

function StatusBadge({ ok, text }: { ok: boolean; text: string }) {
  return (
    <Badge tone={ok ? "ok" : "warn"}>
      {ok ? <CircleCheck className="size-3" /> : <CircleAlert className="size-3" />} {text}
    </Badge>
  );
}
