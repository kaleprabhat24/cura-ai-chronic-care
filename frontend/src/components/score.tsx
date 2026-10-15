"use client";

import { ArrowDownRight, ArrowUpRight, History, Info } from "lucide-react";
import type { Schemas } from "@/lib/api/client";
import { fmtDate, fmtDateTime, humanize, pct } from "@/lib/format";
import { FOLLOW_UP_LABELS, presentScore, tierTone, type ScoreContext } from "@/lib/score";
import { Badge } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

type Prediction = Schemas["PredictionSummary"];
type Factor = Schemas["Factor"];

const PROVENANCE: Record<string, { label: string; tone: "brand" | "neutral" | "info" | "ok" }> = {
  simulated_prediction: { label: "Simulated prediction", tone: "brand" },
  recorded_synthetic_clinical: { label: "Recorded (synthetic)", tone: "neutral" },
  recorded_synthetic_outcome: { label: "Recorded outcome", tone: "info" },
  recorded_outreach_outcome: { label: "Recorded outreach", tone: "ok" },
};

export function ProvenanceTag({ provenance, className }: { provenance: string; className?: string }) {
  const p = PROVENANCE[provenance] ?? { label: humanize(provenance), tone: "neutral" as const };
  return (
    <Badge tone={p.tone} className={cn("text-[11px]", className)}>
      {p.label}
    </Badge>
  );
}

export function TierBadge({ tier, muted }: { tier: string | null | undefined; muted?: boolean }) {
  if (!tier) return <Badge>Unscored</Badge>;
  return (
    <Badge tone={muted ? "neutral" : tierTone(tier)} className="capitalize">
      <span
        className={cn("size-1.5 rounded-full", {
          "bg-tier-high": tier === "high" && !muted,
          "bg-tier-medium": tier === "medium" && !muted,
          "bg-tier-low": tier === "low" || muted,
        })}
      />
      {tier}
    </Badge>
  );
}

export function FollowUpBadge({ ctx }: { ctx: ScoreContext }) {
  const tone =
    ctx.follow_up_status === "overdue_unresolved"
      ? "danger"
      : ctx.follow_up_status === "due_in_grace_window"
        ? "warn"
        : ctx.follow_up_status === "completed"
          ? "ok"
          : "neutral";
  return <Badge tone={tone}>{FOLLOW_UP_LABELS[ctx.follow_up_status]}</Badge>;
}

/** Score cell: a live prediction shows probability + tier; a historical score is shown muted and
 * labelled with its prediction date, never as a fresh prediction. */
export function ScoreCell({ prediction, ctx }: { prediction: Prediction | null | undefined; ctx: ScoreContext }) {
  const view = presentScore(ctx);
  if (!prediction) {
    return (
      <div className="space-y-1">
        <Badge tone="warn">No usable score</Badge>
        {ctx.data_sufficiency === "limited_history" && <p className="text-xs text-ink-subtle">Limited history</p>}
      </div>
    );
  }
  if (view.showProbabilityAsCurrent) {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="tabular text-[15px] font-semibold text-ink">{pct(prediction.probability)}</span>
          <TierBadge tier={prediction.risk_tier} />
        </div>
        <p className="text-xs text-ink-subtle">
          Predicted {fmtDate(prediction.prediction_at)} · {prediction.model_version}
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-ink-muted">
        <History className="size-3.5" />
        <span className="tabular text-sm line-through decoration-ink-subtle/40">{pct(prediction.probability)}</span>
        <TierBadge tier={prediction.risk_tier} muted />
      </div>
      <p className="text-xs text-ink-subtle">
        Historical · predicted {fmtDate(prediction.prediction_at)} ({ctx.score_age_days} d ago)
      </p>
    </div>
  );
}

export function ScoreProvenance({ prediction }: { prediction: Prediction }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
      <dt className="text-ink-subtle">Prediction time</dt>
      <dd className="tabular text-ink">{fmtDateTime(prediction.prediction_at)}</dd>
      <dt className="text-ink-subtle">Scored at</dt>
      <dd className="tabular text-ink">{fmtDateTime(prediction.scored_at)}</dd>
      <dt className="text-ink-subtle">Model</dt>
      <dd className="text-ink">
        {prediction.model_version} · run #{prediction.scoring_run_id}
      </dd>
    </dl>
  );
}

export function ScoreContextNotice({ ctx }: { ctx: ScoreContext }) {
  const view = presentScore(ctx);
  const tones = {
    brand: "border-brand/20 bg-brand-soft text-brand-strong",
    neutral: "border-border bg-surface-muted text-ink-muted",
    danger: "border-danger/20 bg-danger-soft text-danger",
    warn: "border-warn/25 bg-warn-soft text-warn",
  };
  return (
    <div className={cn("flex gap-2 rounded-lg border px-3 py-2 text-[13px]", tones[view.tone])}>
      <Info className="mt-0.5 size-4 shrink-0" />
      <div>
        <p className="font-medium">{view.headline}</p>
        <p className="mt-0.5 opacity-90">{ctx.summary}</p>
      </div>
    </div>
  );
}

/** Model contribution bars (log-odds). Not percentages of risk. */
export function ExplanationFactors({ factors, compact }: { factors: Factor[]; compact?: boolean }) {
  const max = Math.max(...factors.map((f) => Math.abs(f.contribution_log_odds)), 0.0001);
  return (
    <ul className="space-y-2.5">
      {factors.map((f) => {
        const up = f.contribution_log_odds > 0;
        const width = `${Math.max(4, (Math.abs(f.contribution_log_odds) / max) * 100)}%`;
        return (
          <li key={f.feature} className="grid grid-cols-[minmax(0,1fr)_120px] items-center gap-3">
            <div className="min-w-0">
              <p className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                {up ? (
                  <ArrowUpRight className="size-3.5 shrink-0 text-tier-high" />
                ) : (
                  <ArrowDownRight className="size-3.5 shrink-0 text-ok" />
                )}
                <span className="truncate">{f.label}</span>
              </p>
              {!compact && (
                <p className="mt-0.5 pl-5 text-xs text-ink-subtle">
                  Recorded value: <span className="text-ink-muted">{f.value_text}</span>
                  {f.window ? <> · window {f.window}</> : null} · {up ? "raises" : "lowers"} the model&apos;s estimate
                </p>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-muted">
                <div className={cn("h-full rounded-full", up ? "bg-tier-high/70" : "bg-ok/60")} style={{ width }} />
              </div>
              <span className="tabular w-12 text-right text-xs text-ink-muted">
                {up ? "+" : ""}
                {f.contribution_log_odds.toFixed(2)}
              </span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
