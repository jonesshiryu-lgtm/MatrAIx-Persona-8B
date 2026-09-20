import { describe, expect, it } from "vitest";

import { isBatchResumable, type BatchResumeInput } from "../setup/batchResume";

const base: BatchResumeInput = {
  batchJobName: "pg-survey-interrupted",
  batchCancelled: false,
  batchComplete: false,
  launchStatus: null,
  observed: true,
  pendingTrials: 12,
};

describe("isBatchResumable", () => {
  it("offers resume for an attached, observed cohort with no driver", () => {
    expect(isBatchResumable(base)).toBe(true);
    // A coordinator that threw still leaves the cohort intact.
    expect(isBatchResumable({ ...base, launchStatus: "failed" })).toBe(true);
    expect(isBatchResumable({ ...base, launchStatus: "FAILED" })).toBe(true);
  });

  it("never offers resume while the API process is driving the batch", () => {
    expect(isBatchResumable({ ...base, launchStatus: "running" })).toBe(false);
    expect(isBatchResumable({ ...base, launchStatus: "queued" })).toBe(false);
    expect(isBatchResumable({ ...base, launchStatus: "completed" })).toBe(false);
  });

  it("stays hidden without a job, after a stop, once complete, or with nothing pending", () => {
    expect(isBatchResumable({ ...base, batchJobName: null })).toBe(false);
    expect(isBatchResumable({ ...base, batchCancelled: true })).toBe(false);
    expect(isBatchResumable({ ...base, batchComplete: true })).toBe(false);
    expect(isBatchResumable({ ...base, pendingTrials: 0 })).toBe(false);
  });

  it("waits for the first feed response so the button cannot flash", () => {
    expect(isBatchResumable({ ...base, observed: false })).toBe(false);
  });
});
