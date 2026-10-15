"use client";

import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { PageHeader } from "@/components/app-shell";
import { COLORS, SimpleBars, SimpleLines } from "@/components/charts";
import { Stat } from "@/components/stat";
import { ErrorState, InlineNotice, LoadingBlock } from "@/components/states";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { fmtDate, fmtMonth, humanize, num, OUTCOMES, OUTCOME_LABELS, pct } from "@/lib/format";

type Group = { group: string | number; labelled: number; completed_on_time_rate: number };

export default function OutcomesPage() {
  const q = useQuery({ queryKey: ["outcomes"], queryFn: () => unwrap(api.GET("/api/v1/outreach/outcomes")) });

  const monthly = useMemo(() => {
    const rows = new Map<string, { month: string; reached: number; not_reached: number; letter: number }>();
    for (const r of (q.data?.attempts_by_month ?? []) as { month: string; origin: string; group: string; attempts: number }[]) {
      if (r.origin !== "historical_simulation") continue;
      const row = rows.get(r.month) ?? { month: fmtMonth(r.month), reached: 0, not_reached: 0, letter: 0 };
      row[r.group as "reached" | "not_reached" | "letter"] += r.attempts;
      rows.set(r.month, row);
    }
    return [...rows.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, v]) => v);
  }, [q.data]);

  if (q.isLoading) return <LoadingBlock rows={8} />;
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} />;
  const d = q.data!;
  const hist = d.attempts_by_outcome.historical_simulation ?? {};
  const app = d.attempts_by_outcome.app ?? {};
  const outcomeRows = OUTCOMES.map((o) => ({ outcome: OUTCOME_LABELS[o], historical: hist[o] ?? 0, app: app[o] ?? 0 }));
  const byYear = (d.completion_by_year as Group[]).map((g) => ({ year: String(g.group), rate: g.completed_on_time_rate, n: g.labelled }));
  const byProgramme = (d.completion_by_programme as Group[]).map((g) => ({
    programme: g.group === "t2d" ? "Type 2 diabetes" : "Hypertension",
    rate: g.completed_on_time_rate,
  }));
  const exposure = d.completion_by_outreach_exposure as Group[];
  const appTotal = d.attempts_by_origin.app ?? 0;
  const appReached = (app.reached_booked ?? 0) + (app.reached_declined ?? 0) + (app.reached_already_done ?? 0);

  return (
    <>
      <PageHeader
        title="Outcomes analytics"
        description={`Outreach attempts and recorded follow-up completion from persisted records (synthetic, as of ${fmtDate(d.as_of)}).`}
      />
      <InlineNotice tone="warn" className="mb-4">
        <strong>Descriptive statistics, not causal evidence.</strong> Outreach was targeted at patients who were already overdue,
        so completion rates with and without outreach must not be read as the effect of outreach.
      </InlineNotice>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Historical attempts" value={num(d.attempts_by_origin.historical_simulation ?? 0)} />
        <Stat label="Phone contact rate" value={pct(d.phone_contact_rate.historical_simulation ?? null)} hint="Historical, reached ÷ calls" />
        <Stat label="Recorded in Cura AI" value={num(appTotal)} tone="brand" hint={`${num(appReached)} reached`} />
        <Stat label="Open next actions" value={num(d.app_open_next_actions)} tone="warn" hint="Cura AI entries on open follow-ups" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Historical attempts per month</CardTitle>
              <CardDescription>Last 24 months before the cutoff</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={monthly} x="month" stacked height={240}
                        bars={[{ key: "reached", label: "Reached", color: COLORS.brand },
                               { key: "not_reached", label: "Not reached", color: COLORS.medium },
                               { key: "letter", label: "Recall letter", color: COLORS.low }]} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Contact outcomes</CardTitle>
              <CardDescription>All seven outcome categories, by origin. Historical &ldquo;Other&rdquo; includes recall letters.</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={outcomeRows} x="outcome" layout="vertical" height={260}
                        bars={[{ key: "historical", label: "Historical", color: COLORS.low },
                               { key: "app", label: "Cura AI", color: COLORS.brand }]} />
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div>
              <CardTitle>Follow-up completed on time, by prediction year</CardTitle>
              <CardDescription>Labelled (mature, uncensored) episodes; completed within 30 days of the due date</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleLines data={byYear} x="year" yDomain={[0, 1]} yFormatter={(v) => pct(v)} height={230}
                         lines={[{ key: "rate", label: "Completed on time", color: COLORS.brand }]} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>By condition</CardTitle>
              <CardDescription>Completed on time</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={byProgramme} x="programme" yFormatter={(v) => pct(v)} height={230}
                        bars={[{ key: "rate", label: "Completed on time", color: COLORS.brand }]} />
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader>
          <div>
            <CardTitle>Completion by outreach exposure (descriptive)</CardTitle>
            <CardDescription>Not an estimate of the effect of outreach</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          {exposure.map((g) => (
            <div key={String(g.group)} className="rounded-lg border border-border p-4">
              <p className="text-[13px] text-ink-subtle">{humanize(String(g.group))}</p>
              <p className="tabular mt-1 text-2xl font-semibold">{pct(g.completed_on_time_rate)}</p>
              <p className="text-xs text-ink-subtle">{num(g.labelled)} labelled episodes</p>
            </div>
          ))}
          <ul className="list-disc space-y-1 pl-4 text-[12px] text-ink-subtle md:col-span-2">
            {d.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </>
  );
}
