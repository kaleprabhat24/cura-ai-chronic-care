import { describe, expect, it } from "vitest";
import { summariseBundle } from "@/components/fhir-export";
import { presentScore, type ScoreContext } from "./score";

const base: ScoreContext = {
  as_of: "2024-12-31",
  follow_up_status: "upcoming",
  score_status: "predictive",
  outcome_window_ended: false,
  outcome_mature: false,
  days_overdue: 0,
  score_age_days: 20,
  data_sufficiency: "adequate",
  summary: "",
};

describe("score presentation", () => {
  it("shows live predictions as current", () => {
    expect(presentScore(base)).toMatchObject({ kind: "predictive", showProbabilityAsCurrent: true });
    expect(presentScore({ ...base, follow_up_status: "due_in_grace_window", days_overdue: 10 }).showProbabilityAsCurrent).toBe(true);
  });

  it("never presents an overdue, matured episode's score as a current prediction", () => {
    const v = presentScore({ ...base, follow_up_status: "overdue_unresolved", score_status: "historical", outcome_window_ended: true, days_overdue: 400 });
    expect(v.kind).toBe("overdue");
    expect(v.showProbabilityAsCurrent).toBe(false);
    expect(v.headline).toContain("400");
  });

  it("treats completed / censored episodes as historical", () => {
    for (const status of ["completed", "censored"] as const) {
      const v = presentScore({ ...base, follow_up_status: status, score_status: "historical" });
      expect(v).toMatchObject({ kind: "historical", showProbabilityAsCurrent: false });
    }
  });

  it("flags missing scores", () => {
    expect(presentScore({ ...base, score_status: "no_score" })).toMatchObject({ kind: "none", showProbabilityAsCurrent: false });
  });
});

describe("FHIR bundle summary", () => {
  it("counts resources and checks the bundle type", () => {
    const s = summariseBundle({
      resourceType: "Bundle",
      type: "collection",
      entry: [{ resource: { resourceType: "Patient" } }, { resource: { resourceType: "Observation" } }, { resource: { resourceType: "Observation" } }],
    });
    expect(s).toEqual({ ok: true, counts: { Patient: 1, Observation: 2 } });
    expect(summariseBundle({ resourceType: "OperationOutcome" }).ok).toBe(false);
  });
});
