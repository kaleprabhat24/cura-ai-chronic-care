"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, PhoneCall } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { OutreachForm } from "@/components/outreach-form";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/states";
import { Sheet, SheetContent } from "@/components/ui/overlay";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/overlay";
import { Badge, Button, Card, Label, Select, Table, Td, Th } from "@/components/ui/primitives";
import { api, unwrap, type Schemas } from "@/lib/api/client";
import type { paths } from "@/lib/api/schema";
import { fmtDate, fmtDateTime, humanize, num, OUTCOMES, OUTCOME_LABELS } from "@/lib/format";
import { FOLLOW_UP_LABELS } from "@/lib/score";

type ListQuery = NonNullable<paths["/api/v1/outreach-attempts"]["get"]["parameters"]["query"]>;
type Item = Schemas["OutreachListItem"];

const VIEWS = {
  pending: { label: "Pending next actions", query: { pending_next_action: true, origin: "app", sort: "next_action_due" } },
  app: { label: "Recorded in Cura AI", query: { origin: "app" } },
  historical: { label: "Historical (simulated)", query: { origin: "historical_simulation" } },
} as const;

function OutreachView() {
  const [view, setView] = useState<keyof typeof VIEWS>("pending");
  const [outcome, setOutcome] = useState("");
  const [page, setPage] = useState(1);
  const [target, setTarget] = useState<Item | null>(null);
  const query: ListQuery = {
    ...(VIEWS[view].query as ListQuery),
    page,
    page_size: 25,
    ...(outcome ? { outcome: [outcome as NonNullable<ListQuery["outcome"]>[number]] } : {}),
  };
  const list = useQuery({
    queryKey: ["outreach-list", query],
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/api/v1/outreach-attempts", { params: { query } })),
  });
  const today = new Date().toISOString().slice(0, 10);

  return (
    <>
      <PageHeader
        title="Outreach"
        description="Contact attempts and their recorded outcomes. Entries are saved to the database and remain after refresh or restart."
        actions={
          <Button asChild variant="outline">
            <Link href="/worklist">Find follow-ups to contact</Link>
          </Button>
        }
      />
      <div className="flex flex-wrap items-end justify-between gap-3">
        <Tabs value={view} onValueChange={(v) => { setView(v as keyof typeof VIEWS); setPage(1); }}>
          <TabsList>
            {Object.entries(VIEWS).map(([id, v]) => (
              <TabsTrigger key={id} value={id}>
                {v.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <div className="space-y-1.5">
          <Label htmlFor="outcome">Outcome</Label>
          <Select id="outcome" value={outcome} onChange={(e) => { setOutcome(e.target.value); setPage(1); }}>
            <option value="">All outcomes</option>
            {OUTCOMES.map((o) => (
              <option key={o} value={o}>
                {OUTCOME_LABELS[o]}
              </option>
            ))}
          </Select>
        </div>
      </div>

      <Card className="mt-4">
        {list.isLoading ? (
          <div className="p-4">
            <LoadingBlock rows={6} />
          </div>
        ) : list.error ? (
          <ErrorState error={list.error} onRetry={() => list.refetch()} />
        ) : !list.data?.items.length ? (
          <EmptyState
            title={view === "pending" ? "No pending next actions" : "No outreach recorded"}
            hint={view === "pending" ? "Record an attempt with a next action from a patient's page or the worklist." : undefined}
          />
        ) : (
          <>
            <div className="px-4 py-2.5 text-[12px] text-ink-subtle">{num(list.data.total)} attempts</div>
            <Table aria-label="Outreach attempts">
              <thead>
                <tr>
                  <Th>Patient</Th>
                  <Th>Outcome</Th>
                  <Th>Attempted</Th>
                  <Th>Next action</Th>
                  <Th>Follow-up status</Th>
                  <Th>Recorded by</Th>
                  <Th className="sr-only">Actions</Th>
                </tr>
              </thead>
              <tbody>
                {list.data.items.map((o) => {
                  const overdueAction = o.next_action_due && o.next_action_due < today && o.next_action !== "none";
                  return (
                    <tr key={o.id} className="hover:bg-surface-muted/50">
                      <Td>
                        <Link href={`/patients/${o.patient_id}${o.episode_id ? `?episode=${o.episode_id}` : ""}`}
                              className="font-medium hover:text-brand">
                          {o.patient.display_name}
                        </Link>
                        <p className="text-xs text-ink-subtle">{o.patient.synthetic_mrn} · {o.programme.toUpperCase()}</p>
                      </Td>
                      <Td>
                        <p className="text-[13px] font-medium">{OUTCOME_LABELS[o.outcome] ?? o.outcome}</p>
                        <p className="text-xs text-ink-subtle">{humanize(o.channel)}</p>
                        {o.note && <p className="mt-1 max-w-64 text-xs text-ink-muted">{o.note}</p>}
                      </Td>
                      <Td className="tabular text-[13px]">{fmtDateTime(o.attempted_at)}</Td>
                      <Td className="text-[13px]">
                        {o.next_action === "none" ? (
                          <span className="text-ink-subtle">—</span>
                        ) : (
                          <>
                            <p>{humanize(o.next_action)}</p>
                            {o.next_action_due && (
                              <Badge tone={overdueAction ? "danger" : "warn"}>
                                {overdueAction ? "Overdue · " : "Due "}
                                {fmtDate(o.next_action_due)}
                              </Badge>
                            )}
                          </>
                        )}
                      </Td>
                      <Td>{o.follow_up_status ? <Badge>{FOLLOW_UP_LABELS[o.follow_up_status]}</Badge> : "—"}</Td>
                      <Td className="text-[13px]">
                        {o.recorded_by}
                        <p className="text-xs text-ink-subtle">{o.origin === "app" ? "Cura AI" : "Historical"}</p>
                      </Td>
                      <Td>
                        {o.episode_id && o.follow_up_status !== "not_eligible" && (
                          <Button size="sm" variant="outline" onClick={() => setTarget(o)}>
                            <PhoneCall /> Follow up
                          </Button>
                        )}
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
            <div className="flex items-center justify-between px-4 py-3 text-[13px]">
              <span className="text-ink-subtle">
                Page {list.data.page} of {Math.max(1, list.data.total_pages)}
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                  <ChevronLeft /> Previous
                </Button>
                <Button variant="outline" size="sm" disabled={page >= list.data.total_pages} onClick={() => setPage(page + 1)}>
                  Next <ChevronRight />
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>
      <Sheet open={!!target} onOpenChange={(open) => !open && setTarget(null)}>
        {target && target.episode_id && (
          <SheetContent title="Record follow-up outreach"
                        description={`${target.patient.display_name} · ${target.patient.synthetic_mrn}`}>
            <OutreachForm episodeId={target.episode_id} />
          </SheetContent>
        )}
      </Sheet>
    </>
  );
}

export default function OutreachPage() {
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <OutreachView />
    </RoleGate>
  );
}
