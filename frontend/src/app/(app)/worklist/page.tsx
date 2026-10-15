"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { FollowUpBadge, ScoreCell } from "@/components/score";
import { EmptyState, ErrorState, InlineNotice, LoadingBlock } from "@/components/states";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/overlay";
import { Badge, Button, Card, Input, Label, Select, Table, Td, Th } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import type { paths } from "@/lib/api/schema";
import { fmtDate, fmtDateTime, num, OUTCOME_LABELS, relativeDays } from "@/lib/format";

type WorklistQuery = NonNullable<paths["/api/v1/worklist"]["get"]["parameters"]["query"]>;

const QUEUES = [
  { id: "predictive", label: "At risk (predictive)", hint: "Open follow-ups whose outcome window is still running. Ranked by the model's calibrated probability." },
  { id: "overdue", label: "Overdue (operational)", hint: "Not completed by the end of the 30-day outcome window — an observed outcome needing follow-up. Stored scores are historical." },
  { id: "no_score", label: "No usable score", hint: "Open follow-ups without a score in the latest scoring run." },
  { id: "all", label: "All open", hint: "Every open eligible follow-up." },
] as const;

const SORTS = [
  { id: "priority", label: "Model priority" },
  { id: "probability", label: "Probability" },
  { id: "due_date", label: "Due date" },
  { id: "last_contact", label: "Last contact" },
  { id: "patient_name", label: "Patient name" },
] as const;

function useUrlState() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const get = (k: string, d = "") => params.get(k) ?? d;
  const set = (updates: Record<string, string | null>, resetPage = true) => {
    const next = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(updates)) {
      if (v === null || v === "") next.delete(k);
      else next.set(k, v);
    }
    if (resetPage) next.delete("page");
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  };
  return { params, get, set };
}

function WorklistView() {
  const { get, set } = useUrlState();
  const queue = get("queue", "predictive");
  const sort = get("sort", queue === "overdue" ? "due_date" : "priority");
  const page = Number(get("page", "1")) || 1;
  const [search, setSearch] = useState(get("q"));
  useEffect(() => setSearch(get("q")), [get("q")]); // eslint-disable-line react-hooks/exhaustive-deps

  const query = {
    queue: queue as WorklistQuery["queue"],
    sort: sort as WorklistQuery["sort"],
    page,
    page_size: 25,
    ...(get("order") ? { order: get("order") } : {}),
    ...(get("q") ? { q: get("q") } : {}),
    ...(get("tier") ? { tier: get("tier").split(",") } : {}),
    ...(get("condition") ? { condition: get("condition").split(",") } : {}),
    ...(get("due_from") ? { due_from: get("due_from") } : {}),
    ...(get("due_to") ? { due_to: get("due_to") } : {}),
  } as WorklistQuery;
  const list = useQuery({
    queryKey: ["worklist", query],
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/api/v1/worklist", { params: { query } })),
  });
  const queueInfo = QUEUES.find((q) => q.id === queue) ?? QUEUES[0];
  const data = list.data;

  return (
    <>
      <PageHeader
        title="Follow-up worklist"
        description="Prioritised chronic-care follow-ups. Probabilities are simulated predictions; every score shows when it was predicted and by which model."
      />
      <Tabs value={queue} onValueChange={(v) => set({ queue: v, sort: null })}>
        <TabsList>
          {QUEUES.map((q) => (
            <TabsTrigger key={q.id} value={q.id}>
              {q.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <p className="mt-2 text-[13px] text-ink-subtle">{queueInfo.hint}</p>

      <Card className="mt-4">
        <form
          className="flex flex-wrap items-end gap-3 border-b border-border p-4"
          onSubmit={(e) => {
            e.preventDefault();
            set({ q: search.trim() || null });
          }}
        >
          <div className="min-w-56 flex-1 space-y-1.5">
            <Label htmlFor="q">Search</Label>
            <div className="relative">
              <Search className="pointer-events-none absolute top-2.5 left-3 size-4 text-ink-subtle" />
              <Input id="q" className="pl-9" placeholder="Name, MRN or patient id" value={search}
                     onChange={(e) => setSearch(e.target.value)} maxLength={80} />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="condition">Condition</Label>
            <Select id="condition" value={get("condition")} onChange={(e) => set({ condition: e.target.value })}>
              <option value="">All conditions</option>
              <option value="t2d">Type 2 diabetes</option>
              <option value="htn">Hypertension</option>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="tier">Priority tier</Label>
            <Select id="tier" value={get("tier")} onChange={(e) => set({ tier: e.target.value })}>
              <option value="">All tiers</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="high,medium">High + medium</option>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="from">Due from</Label>
            <Input id="from" type="date" value={get("due_from")} onChange={(e) => set({ due_from: e.target.value })} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="to">Due to</Label>
            <Input id="to" type="date" value={get("due_to")} onChange={(e) => set({ due_to: e.target.value })} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="sort">Sort by</Label>
            <Select id="sort" value={sort} onChange={(e) => set({ sort: e.target.value, order: null })}>
              {SORTS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </Select>
          </div>
          <Button type="submit" variant="outline">
            Apply
          </Button>
          <Button type="button" variant="ghost" onClick={() => set({ q: null, tier: null, condition: null, due_from: null, due_to: null, sort: null, order: null })}>
            Reset
          </Button>
        </form>

        {list.isLoading ? (
          <div className="p-4">
            <LoadingBlock rows={8} />
          </div>
        ) : list.error ? (
          <ErrorState error={list.error} onRetry={() => list.refetch()} />
        ) : !data || data.items.length === 0 ? (
          <EmptyState title="No follow-ups match these filters" hint="Try another queue, widen the due-date range or clear the search." />
        ) : (
          <>
            <div className="flex items-center justify-between px-4 py-2.5 text-[12px] text-ink-subtle">
              <span>
                {num(data.total)} follow-ups · as of {fmtDate(data.as_of)} · scoring run #{data.scoring_run_id ?? "—"}
              </span>
              {queue === "overdue" && <Badge tone="danger">Recorded outcomes, not predictions</Badge>}
            </div>
            <Table aria-label="Follow-up worklist">
              <thead>
                <tr>
                  <Th>Patient</Th>
                  <Th>Condition</Th>
                  <Th>Due</Th>
                  <Th>Score</Th>
                  <Th>Leading factor</Th>
                  <Th>Last contact</Th>
                  <Th className="sr-only">Open</Th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((i) => {
                  const f = i.prediction?.top_factor;
                  return (
                    <tr key={i.episode_id} className="hover:bg-surface-muted/50">
                      <Td>
                        <Link href={`/patients/${i.patient.id}?episode=${i.episode_id}`} className="font-medium text-ink hover:text-brand">
                          {i.patient.display_name}
                        </Link>
                        <p className="text-xs text-ink-subtle">
                          {i.patient.synthetic_mrn} · {i.patient.age_years} y · {i.patient.sex}
                        </p>
                      </Td>
                      <Td className="text-[13px]">{i.condition}</Td>
                      <Td>
                        <p className="tabular text-[13px]">{fmtDate(i.due_date)}</p>
                        <p className="mb-1 text-xs text-ink-subtle">{relativeDays(i.days_until_due)}</p>
                        <FollowUpBadge ctx={i.score_context} />
                      </Td>
                      <Td>
                        <ScoreCell prediction={i.prediction} ctx={i.score_context} />
                        {i.prediction && (
                          <p className="mt-1 text-[11px] text-ink-subtle">Scored {fmtDateTime(i.prediction.scored_at)}</p>
                        )}
                      </Td>
                      <Td className="max-w-64">
                        {f ? (
                          <>
                            <p className="text-[13px] text-ink">{f.label}</p>
                            <p className="text-xs text-ink-subtle">
                              {f.value_text} · {f.contribution_log_odds > 0 ? "raises" : "lowers"} estimate
                            </p>
                          </>
                        ) : (
                          <span className="text-xs text-ink-subtle">—</span>
                        )}
                      </Td>
                      <Td className="text-[13px]">
                        {i.last_outreach ? (
                          <>
                            <p>{OUTCOME_LABELS[i.last_outreach.outcome] ?? i.last_outreach.outcome}</p>
                            <p className="text-xs text-ink-subtle">
                              {fmtDate(i.last_outreach.attempted_at)} · {i.outreach_attempts} total
                            </p>
                          </>
                        ) : (
                          <span className="text-xs text-ink-subtle">None</span>
                        )}
                      </Td>
                      <Td>
                        <Button asChild size="sm" variant="outline">
                          <Link href={`/patients/${i.patient.id}?episode=${i.episode_id}`}>Open</Link>
                        </Button>
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
            <div className="flex items-center justify-between px-4 py-3 text-[13px]">
              <span className="text-ink-subtle">
                Page {data.page} of {Math.max(1, data.total_pages)}
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => set({ page: String(page - 1) }, false)}>
                  <ChevronLeft /> Previous
                </Button>
                <Button variant="outline" size="sm" disabled={page >= data.total_pages}
                        onClick={() => set({ page: String(page + 1) }, false)}>
                  Next <ChevronRight />
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
      {queue === "predictive" && (
        <InlineNotice className="mt-4">
          Scores are model predictions on synthetic data — associations, not causes, and not clinical assessments.
        </InlineNotice>
      )}
    </>
  );
}

export default function WorklistPage() {
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <Suspense fallback={<LoadingBlock rows={8} />}>
        <WorklistView />
      </Suspense>
    </RoleGate>
  );
}
