"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Search, UserPlus } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { EmptyState, ErrorState, LoadingBlock } from "@/components/states";
import { Badge, Button, Card, CardContent, Input, Select, Table, Td, Th } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { conditionLabel, fmtDate } from "@/lib/format";

type Origin = "all" | "registered" | "synthetic";

function Patients() {
  const [term, setTerm] = useState("");
  const [q, setQ] = useState("");
  const [origin, setOrigin] = useState<Origin>("all");
  const [page, setPage] = useState(1);
  const sort = origin === "registered" ? "recently_registered" : "name";

  const list = useQuery({
    queryKey: ["patients", q, origin, page],
    placeholderData: keepPreviousData,
    queryFn: () =>
      unwrap(api.GET("/api/v1/patients", { params: { query: { ...(q ? { q } : {}), origin, sort, page, page_size: 25 } } })),
  });

  return (
    <>
      <PageHeader
        title="Patients"
        description="Search the synthetic cohort and patients registered in Cura AI."
        actions={
          <Button asChild>
            <Link href="/patients/new"><UserPlus /> Register patient</Link>
          </Button>
        }
      />
      <Card>
        <CardContent className="pt-5">
          <form
            className="mb-4 flex flex-wrap items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              setPage(1);
              setQ(term.trim());
            }}
          >
            <div className="relative min-w-64 flex-1">
              <Search className="absolute left-3 top-2.5 size-4 text-ink-subtle" />
              <Input className="pl-9" placeholder="Name, MRN or patient id" aria-label="Search patients" maxLength={80}
                     value={term} onChange={(e) => setTerm(e.target.value)} />
            </div>
            <Select aria-label="Patient origin" className="w-56" value={origin}
                    onChange={(e) => { setOrigin(e.target.value as Origin); setPage(1); }}>
              <option value="all">All patients</option>
              <option value="registered">Registered in Cura AI</option>
              <option value="synthetic">Synthetic cohort</option>
            </Select>
            <Button type="submit" variant="outline">Search</Button>
          </form>
          {list.isLoading ? (
            <LoadingBlock rows={8} />
          ) : list.error ? (
            <ErrorState error={list.error} onRetry={() => list.refetch()} />
          ) : !list.data?.items.length ? (
            <EmptyState title="No matching patients" hint={origin === "registered" ? "No patients have been registered in Cura AI yet." : undefined} />
          ) : (
            <>
              <Table>
                <thead>
                  <tr>
                    <Th>Patient</Th>
                    <Th>Conditions</Th>
                    <Th>Registered</Th>
                    <Th>Latest follow-up due</Th>
                  </tr>
                </thead>
                <tbody>
                  {list.data.items.map((i) => (
                    <tr key={i.patient.id} className="hover:bg-surface-muted/40">
                      <Td>
                        <Link href={`/patients/${i.patient.id}`} className="font-medium text-ink hover:text-brand-strong hover:underline">
                          {i.patient.display_name}
                        </Link>
                        <span className="block text-xs text-ink-subtle">
                          {i.patient.synthetic_mrn} · {i.patient.age_years} y · {i.patient.sex}
                        </span>
                        {i.registered_in_app && <Badge tone="info" className="mt-1">Registered in Cura AI</Badge>}
                      </Td>
                      <Td className="text-[13px]">{i.programmes.map(conditionLabel).join(", ") || "—"}</Td>
                      <Td className="tabular text-[13px]">{fmtDate(i.registered_on)}</Td>
                      <Td className="tabular text-[13px]">{fmtDate(i.next_due_date)}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="mt-3 flex items-center justify-between text-[13px] text-ink-subtle">
                <span>{list.data.total} patients</span>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((x) => x - 1)}>Previous</Button>
                  <span>Page {list.data.page} of {list.data.total_pages}</span>
                  <Button variant="outline" size="sm" disabled={page >= list.data.total_pages} onClick={() => setPage((x) => x + 1)}>Next</Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </>
  );
}

export default function PatientsPage() {
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <Patients />
    </RoleGate>
  );
}
