import { useI18n } from "@/i18n/I18nProvider";
import { FOCUS_RING, Sym } from "../cockpitShared";
import { CockpitInlineCount } from "./CockpitCountField";
import { ConfigAnotherRunDialog } from "./ConfigAnotherRunDialog";

export type RunLaunchPhase =
  "idle" | "launching" | "running" | "done" | "error";

export interface RunLaunchBarProps {
  canRun: boolean;
  isBatch: boolean;
  personaCount: number;
  parallelTrials: number;
  onParallelTrialsChange: (value: number) => void;
  isRunning: boolean;
  onRun: () => void;
  error?: string | null;
  /** When set, replaces the run button with a progress bar. */
  runPhase?: RunLaunchPhase;
  progressPct?: number;
  progressLabel?: string;
  progressSublabel?: string;
  onNewRun?: () => void;
  onViewJob?: () => void;
  onCancelRun?: () => void;
  cancelRunBusy?: boolean;
  onDownload?: () => void;
  canDownload?: boolean;
  /** Re-run only the failed trials in-place (batch only). */
  onRetryFailed?: () => void;
  failedCount?: number;
  retryBusy?: boolean;
  /** Continue a batch whose dispatcher died with the API process (batch only). */
  onResumeBatch?: () => void;
  pendingCount?: number;
  resumeBusy?: boolean;
  /** When the live panel already shows a failure card, keep the bar to actions only. */
  compactOnFailure?: boolean;
  /** Open the confirm dialog to background this batch and reset setup. */
  onConfigAnotherRun?: () => void;
  configAnotherOpen?: boolean;
  onConfirmConfigAnother?: () => void;
  onCancelConfigAnother?: () => void;
}

export function RunLaunchBar({
  canRun,
  isBatch,
  personaCount,
  parallelTrials,
  onParallelTrialsChange,
  isRunning,
  onRun,
  error,
  runPhase = "idle",
  progressPct = 0,
  progressLabel,
  progressSublabel,
  onNewRun,
  onViewJob,
  onCancelRun,
  cancelRunBusy = false,
  onDownload,
  canDownload = false,
  onRetryFailed,
  failedCount = 0,
  retryBusy = false,
  onResumeBatch,
  pendingCount = 0,
  resumeBusy = false,
  onConfigAnotherRun,
  configAnotherOpen = false,
  onConfirmConfigAnother,
  onCancelConfigAnother,
}: RunLaunchBarProps) {
  const { t } = useI18n();
  const active = runPhase !== "idle";
  const failed = runPhase === "error";
  const done = runPhase === "done";
  const pct =
    runPhase === "launching" && progressPct <= 0
      ? 12
      : Math.max(0, Math.min(100, progressPct));
  const parallelMax = Math.max(1, personaCount);

  return (
    <div className="glass-panel-strong w-full shrink-0 rounded-xl px-4 py-3 sm:px-5">
      {error && (
        <p className="mb-2 w-full rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-[13px] text-danger">
          {error}
        </p>
      )}

      {active ? (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
            <div className="flex min-w-0 grow basis-56 items-center gap-2.5">
              {failed ? (
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-danger/12">
                  <Sym
                    name="error"
                    fill={1}
                    size={18}
                    className="text-danger"
                  />
                </span>
              ) : done ? (
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary/12">
                  <Sym
                    name="check_circle"
                    fill={1}
                    size={18}
                    className="text-secondary"
                  />
                </span>
              ) : (
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <Sym
                    name="autorenew"
                    size={18}
                    className="animate-rb-spin text-primary"
                  />
                </span>
              )}
              <div className="min-w-0">
                <p className="truncate font-display text-[15px] font-semibold leading-tight text-text-main">
                  {progressLabel ??
                    (runPhase === "launching"
                      ? t("cockpitSetup.run.launching")
                      : isBatch
                        ? t("cockpitSetup.run.batch")
                        : t("cockpitSetup.run.runningSimulation"))}
                </p>
                {progressSublabel && (
                  <p className="mt-0.5 truncate text-[12px] text-text-dim">
                    {progressSublabel}
                  </p>
                )}
              </div>
            </div>

            {/* Below sm the actions drop to their own full-width row and share it
                evenly — otherwise they squeeze the progress label to "0 of 200…". */}
            <div className="flex w-full flex-wrap items-center gap-2 sm:ml-auto sm:w-auto sm:shrink-0 sm:justify-end">
              {onCancelRun && !done && !failed && (
                <button
                  type="button"
                  onClick={onCancelRun}
                  disabled={cancelRunBusy}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-danger/35 bg-danger/8 px-3.5 py-2 text-[14px] font-medium text-danger transition hover:border-danger/50 hover:bg-danger/14 active:scale-[0.98] disabled:opacity-50 sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym name="stop_circle" size={16} />
                  {cancelRunBusy
                    ? t("cockpitSetup.run.stopping")
                    : isBatch
                      ? t("cockpitSetup.run.stopBatch")
                      : t("cockpitSetup.run.stopRun")}
                </button>
              )}
              {onConfigAnotherRun && isBatch && (
                <button
                  type="button"
                  onClick={onConfigAnotherRun}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-outline/55 bg-transparent px-3.5 py-2 text-[14px] font-medium text-text-variant transition hover:border-outline hover:bg-surface-low hover:text-text-main active:scale-[0.98] sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym name="tune" size={16} />
                  {t("cockpitSetup.run.configAnother")}
                </button>
              )}
              {onRetryFailed &&
                isBatch &&
                (done || failed) &&
                failedCount > 0 && (
                  <button
                    type="button"
                    onClick={onRetryFailed}
                    disabled={retryBusy}
                    className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-[#e5c07b]/45 bg-[#e5c07b]/10 px-3.5 py-2 text-[14px] font-medium text-[#e5c07b] transition hover:border-[#e5c07b]/60 hover:bg-[#e5c07b]/16 active:scale-[0.98] disabled:opacity-50 sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                  >
                    <Sym
                      name="replay"
                      size={16}
                      className={retryBusy ? "animate-rb-spin" : undefined}
                    />
                    {retryBusy
                      ? t("cockpitSetup.run.retrying")
                      : t("cockpitSetup.run.retryFailed", {
                          count: failedCount,
                        })}
                  </button>
                )}
              {onResumeBatch && isBatch && pendingCount > 0 && (
                <button
                  type="button"
                  onClick={onResumeBatch}
                  disabled={resumeBusy}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-primary/45 bg-primary/10 px-3.5 py-2 text-[14px] font-medium text-primary transition hover:border-primary/60 hover:bg-primary/16 active:scale-[0.98] disabled:opacity-50 sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym
                    name="play_circle"
                    size={16}
                    className={resumeBusy ? "animate-rb-spin" : undefined}
                  />
                  {resumeBusy
                    ? t("cockpitSetup.run.resuming")
                    : t("cockpitSetup.run.resumeBatch", {
                        count: pendingCount,
                      })}
                </button>
              )}
              {onDownload && (done || failed) && !onViewJob && (
                <button
                  type="button"
                  onClick={onDownload}
                  disabled={!canDownload}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-outline/60 bg-surface/40 px-3.5 py-2 text-[14px] font-medium text-text-variant backdrop-blur-sm transition hover:border-outline hover:bg-surface-high hover:text-text-main active:scale-[0.98] disabled:opacity-50 sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym name="download" size={16} />
                  {t("cockpitSetup.run.download")}
                </button>
              )}
              {onViewJob && (done || failed) && (
                <button
                  type="button"
                  onClick={onViewJob}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg bg-primary px-4 py-2 font-display text-[14px] font-semibold text-on-primary shadow-[0_2px_10px_-4px_rgb(0_0_0/0.45)] transition hover:bg-primary-dim active:scale-[0.98] sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym name="open_in_new" size={16} />
                  {isBatch
                    ? t("cockpitSetup.run.viewJob")
                    : t("cockpitSetup.run.viewTrial")}
                </button>
              )}
              {onNewRun && (done || failed) && !(isBatch && onConfigAnotherRun) && (
                <button
                  type="button"
                  onClick={onNewRun}
                  className={`inline-flex min-w-[8.5rem] flex-1 items-center justify-center gap-1.5 rounded-lg border border-outline/55 bg-transparent px-3.5 py-2 text-[14px] font-medium text-text-dim transition hover:border-outline hover:bg-surface-low hover:text-text-variant active:scale-[0.98] sm:min-w-0 sm:flex-none ${FOCUS_RING}`}
                >
                  <Sym name="restart_alt" size={16} />
                  {t("cockpitSetup.run.reset")}
                </button>
              )}
            </div>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-field/80">
            <div
              className={`h-full rounded-full transition-[width] duration-500 ${
                failed ? "bg-danger" : done ? "bg-secondary" : "bg-primary"
              } ${runPhase === "launching" ? "animate-pulse" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      ) : (
        <>
          <div className="flex w-full flex-col items-center gap-3 sm:flex-row sm:justify-center">
            <button
              type="button"
              disabled={!canRun || isRunning}
              onClick={onRun}
              className={`glow inline-flex w-full min-w-[200px] items-center justify-center gap-2 rounded-lg bg-primary px-6 py-3.5 font-display text-[16px] font-bold text-on-primary transition hover:bg-primary-dim disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto ${FOCUS_RING}`}
            >
              <Sym
                name={isBatch ? "rocket_launch" : "play_arrow"}
                fill={1}
                size={22}
              />
              {isRunning
                ? t("cockpitSetup.run.launching")
                : isBatch
                  ? t("cockpitSetup.run.runBatch", { count: personaCount })
                  : t("cockpitSetup.run.runLive")}
            </button>
            {isBatch && personaCount > 1 && (
              <CockpitInlineCount
                label={t("cockpitSetup.run.parallel")}
                value={Math.min(parallelTrials, parallelMax)}
                onChange={onParallelTrialsChange}
                min={1}
                max={parallelMax}
                disabled={isRunning}
                hint={t("cockpitSetup.run.atMost", { count: parallelMax })}
              />
            )}
          </div>
          <p className="mt-2 text-center text-[12px] text-text-dim">
            {isBatch
              ? t("cockpitSetup.run.batchHint")
              : t("cockpitSetup.run.liveHint")}
          </p>
        </>
      )}
      <ConfigAnotherRunDialog
        open={configAnotherOpen}
        onCancel={onCancelConfigAnother ?? (() => undefined)}
        onConfirm={onConfirmConfigAnother ?? (() => undefined)}
      />
    </div>
  );
}
