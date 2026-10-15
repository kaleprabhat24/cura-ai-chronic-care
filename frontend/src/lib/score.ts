/** How a score is presented, driven entirely by the backend's ScoreContext (unit-tested). A
 * probability is only presented as a current prediction when score_status === "predictive". */

import type { Schemas } from "@/lib/api/client";

export type ScoreContext = Schemas["ScoreContext"];

export type ScorePresentation = {
  kind: "predictive" | "historical" | "overdue" | "none";
  headline: string;
  showProbabilityAsCurrent: boolean;
  tone: "brand" | "neutral" | "danger" | "warn";
};

export function presentScore(ctx: ScoreContext): ScorePresentation {
  if (ctx.follow_up_status === "overdue_unresolved") {
    return {
      kind: "overdue",
      headline: `Overdue ${ctx.days_overdue} d — not completed within the outcome window`,
      showProbabilityAsCurrent: false,
      tone: "danger",
    };
  }
  if (ctx.score_status === "predictive") {
    return {
      kind: "predictive",
      headline: ctx.follow_up_status === "due_in_grace_window" ? "Predictive — due date passed, window open" : "Predictive",
      showProbabilityAsCurrent: true,
      tone: "brand",
    };
  }
  if (ctx.score_status === "historical") {
    return { kind: "historical", headline: "Historical score (outcome known)", showProbabilityAsCurrent: false, tone: "neutral" };
  }
  return { kind: "none", headline: "No usable score", showProbabilityAsCurrent: false, tone: "warn" };
}

export const FOLLOW_UP_LABELS: Record<ScoreContext["follow_up_status"], string> = {
  upcoming: "Upcoming",
  due_in_grace_window: "Due — window open",
  overdue_unresolved: "Overdue — unresolved",
  completed: "Completed",
  censored: "Left care",
  not_eligible: "Not eligible",
};

export function tierTone(tier: string | null | undefined): "high" | "medium" | "low" | "neutral" {
  return tier === "high" || tier === "medium" || tier === "low" ? tier : "neutral";
}
