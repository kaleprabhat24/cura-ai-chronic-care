"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CircleCheck, CircleX } from "lucide-react";
import { useState } from "react";
import { PageHeader } from "@/components/app-shell";
import { CalibrationPlot, COLORS, SimpleBars } from "@/components/charts";
import { Stat } from "@/components/stat";
import { ErrorState, InlineNotice, LoadingBlock } from "@/components/states";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Table, Td, Th } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { hasRole, useAuth } from "@/lib/auth/auth-context";
import { fmtDate, fmtDateTime, humanize, num, pct } from "@/lib/format";

const NAMES: Record<string, string> = {
  prevalence_baseline: "Prevalence baseline",
  attendance_heuristic: "Attendance heuristic",
  logistic_regression: "Logistic regression",
  xgboost: "XGBoost",
};
type Metrics = { pr_auc: number; roc_auc: number; brier: number; ece: number; calibration_slope: number;
  prevalence: number; n: number; capacity: Record<string, { precision: number; recall: number; lift: number }> };

export default function ModelPage() {
  const { me } = useAuth();
  const analytics = hasRole(me, "analyst", "admin");
  const [doc, setDoc] = useState<"card" | "report" | null>(null);
  const model = useQuery({ queryKey: ["model"], queryFn: () => unwrap(api.GET("/api/v1/models/current")) });
  const docs = useQuery({ queryKey: ["model-docs"], queryFn: () => unwrap(api.GET("/api/v1/models/current/documents")) });
  const dq = useQuery({ queryKey: ["data-quality"], enabled: analytics, queryFn: () => unwrap(api.GET("/api/v1/data-quality")) });
  const runs = useQuery({
    queryKey: ["scoring-runs"],
    enabled: analytics,
    queryFn: () => unwrap(api.GET("/api/v1/scoring-runs", { params: { query: { page_size: 10 } } })),
  });
  const dash = useQuery({ queryKey: ["dashboard"], queryFn: () => unwrap(api.GET("/api/v1/dashboard/summary")) });

  if (model.isLoading) return <LoadingBlock rows={8} />;
  if (model.error) return <ErrorState error={model.error} onRetry={() => model.refetch()} />;
  const m = model.data!;
  const test = m.test_metrics as Record<string, Metrics>;
  const ci = m.test_intervals as Record<string, Record<string, [number, number]>>;
  const cap = "0.15";
  const sel = m.model_type;
  const comparison = Object.keys(NAMES).map((k) => ({ model: NAMES[k], pr: test[k]?.pr_auc ?? 0, roc: test[k]?.roc_auc ?? 0 }));
  const cal = (m.calibration_table as { mean_predicted: number; observed_rate: number; n: number }[]).map((r) => ({
    predicted: r.mean_predicted, observed: r.observed_rate, n: r.n,
  }));
  const importance = (m.feature_importance as { label: string; mean_abs_log_odds: number }[]).slice(0, 10)
    .map((f) => ({ feature: f.label, value: f.mean_abs_log_odds }));
  const selection = m.selection as { rule: string; xgboost_minus_lr_val_pr_auc: number; xgboost_minus_lr_val_pr_auc_ci?: [number, number] };
  const run = dash.data?.latest_scoring_run;

  return (
    <>
      <PageHeader title="Model & data health" description="The model that produced the stored scores, its evaluation and the state of the loaded data." />
      <div className="mb-4 flex gap-3 rounded-lg border border-warn/30 bg-warn-soft px-4 py-3 text-[13px] text-warn" role="note">
        <AlertTriangle className="mt-0.5 size-4 shrink-0" />
        <p>
          <strong>Evaluated on synthetic data only — not clinically validated.</strong> {m.disclaimer} The prediction target is: {m.prediction_target}.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Model version" value={<span className="text-[20px]">{m.model_version}</span>} hint={`${NAMES[sel]} · ${m.calibration_method} calibration`} />
        <Stat label="Test PR-AUC" value={test[sel].pr_auc.toFixed(3)} tone="brand"
              hint={`95% CI ${ci[sel].pr_auc[0].toFixed(3)}–${ci[sel].pr_auc[1].toFixed(3)} · prevalence ${pct(test[sel].prevalence)}`} />
        <Stat label="Precision at 15% capacity" value={pct(test[sel].capacity[cap].precision)}
              hint={`Recall ${pct(test[sel].capacity[cap].recall)} · lift ${test[sel].capacity[cap].lift.toFixed(2)}×`} />
        <Stat label="Risk-tier thresholds" value={<span className="text-[20px]">≥{pct(m.tier_thresholds.high)} / ≥{pct(m.tier_thresholds.medium)}</span>}
              hint="High / medium, fixed on validation data" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Model comparison (test set)</CardTitle>
              <CardDescription>
                {num(test[sel].n)} held-out patient episodes · patient-level bootstrap intervals
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <thead>
                <tr>
                  <Th>Model</Th>
                  <Th>PR-AUC</Th>
                  <Th>ROC-AUC</Th>
                  <Th>Brier</Th>
                  <Th>Prec@15%</Th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(NAMES).map((k) => (
                  <tr key={k} className={k === sel ? "bg-brand-soft/50" : undefined}>
                    <Td className="text-[13px] font-medium">
                      {NAMES[k]} {k === sel && <Badge tone="brand">selected</Badge>}
                    </Td>
                    <Td className="tabular text-[13px]">
                      {test[k].pr_auc.toFixed(3)}
                      <span className="block text-[11px] text-ink-subtle">{ci[k].pr_auc[0].toFixed(3)}–{ci[k].pr_auc[1].toFixed(3)}</span>
                    </Td>
                    <Td className="tabular text-[13px]">{test[k].roc_auc.toFixed(3)}</Td>
                    <Td className="tabular text-[13px]">{test[k].brier.toFixed(3)}</Td>
                    <Td className="tabular text-[13px]">{pct(test[k].capacity[cap].precision)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="mt-3 text-[12px] text-ink-subtle">
              Selection rule (validation only): {selection.rule}. Observed XGBoost − LR validation PR-AUC{" "}
              {selection.xgboost_minus_lr_val_pr_auc.toFixed(4)}
              {selection.xgboost_minus_lr_val_pr_auc_ci &&
                ` [${selection.xgboost_minus_lr_val_pr_auc_ci[0].toFixed(4)}, ${selection.xgboost_minus_lr_val_pr_auc_ci[1].toFixed(4)}]`}
              .
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>PR-AUC by model</CardTitle>
              <CardDescription>Higher is better; prevalence baseline ≈ event rate</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={comparison} x="model" height={250} yFormatter={(v) => v.toFixed(2)}
                        bars={[{ key: "pr", label: "PR-AUC", color: COLORS.brand }, { key: "roc", label: "ROC-AUC", color: COLORS.low }]} />
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Calibration (test set)</CardTitle>
              <CardDescription>
                Decile mean predicted vs observed · ECE {test[sel].ece.toFixed(3)} · slope {test[sel].calibration_slope.toFixed(2)}
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <CalibrationPlot points={cal} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Global feature importance</CardTitle>
              <CardDescription>Mean |contribution| in log-odds on the test set (not % of risk)</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <SimpleBars data={importance} x="feature" layout="vertical" height={300} yFormatter={(v) => v.toFixed(2)}
                        bars={[{ key: "value", label: "Mean |contribution|", color: COLORS.brand }]} />
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Risk tiers on the test set</CardTitle>
              <CardDescription>Observed incomplete rate per tier (recorded outcomes)</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <thead>
                <tr>
                  <Th>Tier</Th>
                  <Th>Episodes</Th>
                  <Th>Share</Th>
                  <Th>Observed incomplete</Th>
                </tr>
              </thead>
              <tbody>
                {(m.test_tiers as { tier: string; n: number; share: number; observed_incomplete_rate: number }[]).map((t) => (
                  <tr key={t.tier}>
                    <Td className="capitalize">{t.tier}</Td>
                    <Td className="tabular">{num(t.n)}</Td>
                    <Td className="tabular">{pct(t.share)}</Td>
                    <Td className="tabular">{pct(t.observed_incomplete_rate)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            {run && (
              <p className="mt-3 text-[12px] text-ink-subtle">
                Latest scoring run #{run.id}: {Object.entries(run.tier_counts).map(([k, v]) => `${num(v)} ${k}`).join(" · ")} — scored{" "}
                {fmtDateTime(run.finished_at)} on data cut {fmtDate(run.data_cutoff)}.
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Evaluation documents</CardTitle>
              <CardDescription>Generated by the training pipeline</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Button variant={doc === "card" ? "default" : "outline"} size="sm" onClick={() => setDoc(doc === "card" ? null : "card")}
                      disabled={!docs.data?.model_card_markdown}>
                Model card
              </Button>
              <Button variant={doc === "report" ? "default" : "outline"} size="sm" onClick={() => setDoc(doc === "report" ? null : "report")}
                      disabled={!docs.data?.model_report_markdown}>
                Evaluation report
              </Button>
            </div>
            {doc && (
              <pre className="max-h-96 overflow-auto rounded-lg border border-border bg-surface-muted p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-ink-muted">
                {doc === "card" ? docs.data?.model_card_markdown : docs.data?.model_report_markdown}
              </pre>
            )}
            <p className="text-[12px] text-ink-subtle">
              Libraries: {Object.entries(m.library_versions).map(([k, v]) => `${k} ${v}`).join(", ")}.
            </p>
          </CardContent>
        </Card>
      </div>

      {analytics ? (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Data quality &amp; freshness</CardTitle>
                <CardDescription>
                  {dq.data?.dataset
                    ? `Dataset ${dq.data.dataset.dataset_version} · cutoff ${fmtDate(String(dq.data.dataset.data_cutoff))} · loaded ${fmtDateTime(String(dq.data.dataset.loaded_at))}`
                    : "Consistency between dataset files, database, model and scores"}
                </CardDescription>
              </div>
              {dq.data && <Badge tone={dq.data.status === "ok" ? "ok" : "warn"}>{dq.data.status}</Badge>}
            </CardHeader>
            <CardContent>
              {dq.isLoading ? (
                <LoadingBlock rows={5} />
              ) : dq.error ? (
                <ErrorState error={dq.error} />
              ) : (
                <ul className="space-y-1.5 text-[13px]">
                  {dq.data!.checks.map((c) => (
                    <li key={String(c.check)} className="flex items-start gap-2">
                      {c.ok ? <CircleCheck className="mt-0.5 size-4 text-ok" /> : <CircleX className="mt-0.5 size-4 text-danger" />}
                      <span>
                        <span className="font-medium">{humanize(String(c.check))}</span>
                        <span className="block text-xs text-ink-subtle">{String(c.detail)}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Feature missingness</CardTitle>
                <CardDescription>Share of test episodes with the value not recorded</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {dq.data && (
                <SimpleBars
                  data={Object.entries(dq.data.feature_missingness).sort(([, a], [, b]) => b - a).map(([k, v]) => ({ feature: humanize(k), share: v }))}
                  x="feature" layout="vertical" height={300} yFormatter={(v) => pct(v)}
                  bars={[{ key: "share", label: "Missing", color: COLORS.medium }]}
                />
              )}
            </CardContent>
          </Card>
          <Card className="lg:col-span-2">
            <CardHeader>
              <div>
                <CardTitle>Scoring runs</CardTitle>
                <CardDescription>Every run is kept as an immutable snapshot</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              {runs.data && (
                <Table>
                  <thead>
                    <tr>
                      <Th>Run</Th>
                      <Th>Model</Th>
                      <Th>Scope</Th>
                      <Th>Episodes</Th>
                      <Th>Tiers</Th>
                      <Th>Finished</Th>
                      <Th>Data fingerprint</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.items.map((r) => (
                      <tr key={r.id}>
                        <Td>#{r.id}</Td>
                        <Td className="text-[13px]">{r.model_version}</Td>
                        <Td className="text-[13px]">{r.scope}</Td>
                        <Td className="tabular">{num(r.n_scored)}</Td>
                        <Td className="text-[12px]">{Object.entries(r.tier_counts).map(([k, v]) => `${k} ${num(v)}`).join(" · ")}</Td>
                        <Td className="text-[12px]">{fmtDateTime(r.finished_at)}</Td>
                        <Td className="font-mono text-[11px]">{r.data_fingerprint.slice(0, 16)}…</Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      ) : (
        <InlineNotice className="mt-4">Detailed data-quality checks and scoring-run history are available to analyst and admin roles.</InlineNotice>
      )}
    </>
  );
}
