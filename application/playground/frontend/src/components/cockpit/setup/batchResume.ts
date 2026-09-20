/**
 * Decide whether the attached batch can be resumed from the workspace.
 *
 * Local cohorts are dispatched by the API process itself (see
 * `backend/service/local_distributed_harbor.py`), so a backend restart or crash
 * leaves the remaining trials undispatched forever while every artifact on disk
 * still looks fine. The feed stops reporting a launch record in exactly that
 * window, so "no driver + unfinished + job attached" is the resumable state.
 *
 * The inverse matters just as much: an actively driven batch also has pending
 * trials, and offering resume there would dispatch the same cohort twice.
 */
export interface BatchResumeInput {
  batchJobName: string | null;
  batchCancelled: boolean;
  batchComplete: boolean;
  /** Launch status from the live/status feed; null = no driver in this process. */
  launchStatus: string | null;
  /** The feed answered at least once — avoids a flash before the first poll. */
  observed: boolean;
  pendingTrials: number;
}

export function isBatchResumable({
  batchJobName,
  batchCancelled,
  batchComplete,
  launchStatus,
  observed,
  pendingTrials,
}: BatchResumeInput): boolean {
  if (!batchJobName || batchCancelled || batchComplete) return false;
  if (!observed || pendingTrials <= 0) return false;
  const status = (launchStatus ?? "").trim().toLowerCase();
  // No record at all = restarted API; "failed" = the coordinator thread threw.
  // Both leave the cohort intact and resumable. "queued"/"running" is a live
  // driver, "completed" means the dispatcher already worked through the cohort.
  return status === "" || status === "failed";
}
