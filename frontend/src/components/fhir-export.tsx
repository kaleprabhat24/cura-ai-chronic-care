"use client";

import { Download } from "lucide-react";
import { useState } from "react";
import { api, ApiError, unwrap } from "@/lib/api/client";
import { Button } from "@/components/ui/primitives";
import { InlineNotice, Spinner } from "@/components/states";

type Bundle = { resourceType?: string; type?: string; entry?: { resource?: { resourceType?: string } }[] };

/** Summarises a downloaded bundle so the user sees what was exported (and that it parsed). */
export function summariseBundle(bundle: Bundle): { ok: boolean; counts: Record<string, number> } {
  const counts: Record<string, number> = {};
  for (const e of bundle.entry ?? []) {
    const t = e.resource?.resourceType ?? "unknown";
    counts[t] = (counts[t] ?? 0) + 1;
  }
  return { ok: bundle.resourceType === "Bundle" && bundle.type === "collection" && !!bundle.entry?.length, counts };
}

export function FhirExportButton({ patientId, compact }: { patientId: string; compact?: boolean }) {
  const [state, setState] = useState<{ status: "idle" | "busy" | "done" | "error"; text?: string }>({ status: "idle" });

  async function exportBundle() {
    setState({ status: "busy" });
    try {
      const text = await unwrap(
        api.GET("/api/v1/patients/{patient_id}/fhir", { params: { path: { patient_id: patientId } }, parseAs: "text" }),
      );
      const bundle = JSON.parse(text as unknown as string) as Bundle;
      const { ok, counts } = summariseBundle(bundle);
      if (!ok) throw new Error("The server returned an unexpected document.");
      const blob = new Blob([text as unknown as string], { type: "application/fhir+json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `cura-ai-${patientId}-fhir-bundle.json`;
      a.click();
      URL.revokeObjectURL(url);
      const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(", ");
      setState({ status: "done", text: `Downloaded ${bundle.entry!.length} resources (${summary}).` });
    } catch (e) {
      setState({ status: "error", text: e instanceof ApiError || e instanceof Error ? e.message : "Export failed." });
    }
  }

  return (
    <div className="flex flex-col items-end gap-2">
      <Button variant="outline" onClick={exportBundle} disabled={state.status === "busy"}>
        {state.status === "busy" ? <Spinner label="Exporting…" /> : <><Download /> {compact ? "FHIR" : "Export FHIR bundle"}</>}
      </Button>
      {state.status === "done" && <InlineNotice tone="ok" className="max-w-md">{state.text}</InlineNotice>}
      {state.status === "error" && <InlineNotice tone="danger" className="max-w-md">{state.text}</InlineNotice>}
    </div>
  );
}
