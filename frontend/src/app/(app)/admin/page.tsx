"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { useState } from "react";
import { PageHeader, RoleGate } from "@/components/app-shell";
import { FhirExportButton } from "@/components/fhir-export";
import { EmptyState, ErrorState, InlineNotice, LoadingBlock } from "@/components/states";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Input, Label, Select, Table, Td, Th } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { hasRole, useAuth } from "@/lib/auth/auth-context";
import { fmtDateTime, num } from "@/lib/format";

const ACTIONS = ["", "outreach.create", "scoring.run", "dataset.seed", "fhir.export", "user.add", "user.update"];

function FhirExportPanel() {
  const [term, setTerm] = useState("");
  const [submitted, setSubmitted] = useState("");
  const results = useQuery({
    queryKey: ["patient-search", submitted],
    enabled: submitted.length > 0,
    queryFn: () => unwrap(api.GET("/api/v1/worklist", { params: { query: { q: submitted, status: "all", page_size: 10 } } })),
  });
  const patients = new Map((results.data?.items ?? []).map((i) => [i.patient.id, i.patient]));
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle>FHIR R4-oriented export</CardTitle>
          <CardDescription>
            Download a synthetic patient&apos;s record as a JSON Bundle (Patient, Condition, Encounter, Appointment, Observation,
            MedicationStatement, RiskAssessment, Communication). Exports are audited.
          </CardDescription>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <form className="flex items-end gap-2" onSubmit={(e) => { e.preventDefault(); setSubmitted(term.trim()); }}>
          <div className="flex-1 space-y-1.5">
            <Label htmlFor="patient">Find a patient</Label>
            <Input id="patient" placeholder="Name, MRN (e.g. SYN0000831) or patient id" value={term} maxLength={80}
                   onChange={(e) => setTerm(e.target.value)} />
          </div>
          <Button type="submit" variant="outline" disabled={!term.trim()}>
            <Search /> Search
          </Button>
        </form>
        {results.isLoading && <LoadingBlock rows={2} />}
        {results.error && <ErrorState error={results.error} />}
        {results.data && patients.size === 0 && <EmptyState title="No matching patients" />}
        <ul className="divide-y divide-border">
          {[...patients.values()].map((p) => (
            <li key={p.id} className="flex flex-wrap items-center justify-between gap-3 py-3">
              <span className="text-[13px]">
                <span className="font-medium">{p.display_name}</span>
                <span className="block text-xs text-ink-subtle">{p.synthetic_mrn} · {p.id}</span>
              </span>
              <FhirExportButton patientId={p.id} compact />
            </li>
          ))}
        </ul>
        <InlineNotice>
          Structure checked with automated R4 rules and the fhir.resources R4B models. Not validated with the official HL7 FHIR
          validator, so formal R4 conformance is not claimed.
        </InlineNotice>
      </CardContent>
    </Card>
  );
}

function AuditPanel() {
  const [action, setAction] = useState("");
  const [page, setPage] = useState(1);
  const audit = useQuery({
    queryKey: ["audit", action, page],
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/api/v1/audit-events", { params: { query: { page, page_size: 20, ...(action ? { action } : {}) } } })),
  });
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle>Audit log</CardTitle>
          <CardDescription>Persistent record of seeding, scoring, outreach writes, exports and user changes</CardDescription>
        </div>
        <Select aria-label="Filter by action" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }} className="w-48">
          {ACTIONS.map((a) => (
            <option key={a} value={a}>
              {a || "All actions"}
            </option>
          ))}
        </Select>
      </CardHeader>
      <CardContent>
        {audit.isLoading ? (
          <LoadingBlock rows={5} />
        ) : audit.error ? (
          <ErrorState error={audit.error} />
        ) : !audit.data?.items.length ? (
          <EmptyState title="No audit events" />
        ) : (
          <>
            <Table>
              <thead>
                <tr>
                  <Th>When</Th>
                  <Th>Actor</Th>
                  <Th>Action</Th>
                  <Th>Entity</Th>
                  <Th>Details</Th>
                </tr>
              </thead>
              <tbody>
                {audit.data.items.map((e) => (
                  <tr key={e.id}>
                    <Td className="text-[12px]">{fmtDateTime(e.occurred_at)}</Td>
                    <Td className="text-[13px]">
                      {e.actor_id}
                      <span className="block text-[11px] text-ink-subtle">{e.actor_role}</span>
                    </Td>
                    <Td><Badge tone="brand">{e.action}</Badge></Td>
                    <Td className="font-mono text-[11px]">{e.entity_type}/{e.entity_id}</Td>
                    <Td className="max-w-md truncate font-mono text-[11px] text-ink-subtle" title={JSON.stringify(e.details)}>
                      {JSON.stringify(e.details)}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <div className="flex items-center justify-between pt-3 text-[13px]">
              <span className="text-ink-subtle">{num(audit.data.total)} events</span>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" disabled={page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft /> Previous</Button>
                <Button size="sm" variant="outline" disabled={page >= audit.data.total_pages} onClick={() => setPage(page + 1)}>Next <ChevronRight /></Button>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function UsersPanel() {
  const users = useQuery({ queryKey: ["users"], queryFn: () => unwrap(api.GET("/api/v1/admin/users")) });
  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle>Users and roles</CardTitle>
          <CardDescription>
            Provisioned with <code className="font-mono text-[12px]">cura-api users add EMAIL --role ROLE</code>. Roles always come from this
            table, never from the browser.
          </CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        {users.isLoading ? (
          <LoadingBlock rows={3} />
        ) : users.error ? (
          <ErrorState error={users.error} />
        ) : !users.data?.length ? (
          <EmptyState title="No users provisioned" hint="Needed for Supabase sign-in. Development sign-in does not use this table." />
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Email</Th>
                <Th>Role</Th>
                <Th>Status</Th>
                <Th>Linked to Supabase</Th>
                <Th>Last sign-in</Th>
              </tr>
            </thead>
            <tbody>
              {users.data.map((u) => (
                <tr key={u.id}>
                  <Td className="text-[13px]">{u.email}</Td>
                  <Td className="capitalize">{u.role}</Td>
                  <Td>{u.is_active ? <Badge tone="ok">active</Badge> : <Badge tone="danger">inactive</Badge>}</Td>
                  <Td>{u.linked_to_auth_identity ? "yes" : "not yet"}</Td>
                  <Td className="text-[12px]">{fmtDateTime(u.last_sign_in_at)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

export default function AdminPage() {
  const { me } = useAuth();
  const admin = hasRole(me, "admin");
  return (
    <RoleGate roles={["coordinator", "admin"]}>
      <PageHeader title="Administration & export" description="FHIR export, audit history and user roles. Every action here is authorised by the API." />
      <div className="space-y-4">
        <FhirExportPanel />
        {admin ? (
          <>
            <AuditPanel />
            <UsersPanel />
          </>
        ) : (
          <InlineNotice>The audit log and user list are available to administrators.</InlineNotice>
        )}
      </div>
    </RoleGate>
  );
}
