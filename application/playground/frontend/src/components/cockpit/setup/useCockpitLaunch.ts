import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api, ApiError } from "@/lib/api";
import type { HarborCockpitTaskKind } from "@/lib/harborCockpitMappers";
import type { ConfigOptionsResponse } from "@/lib/types";

import { buildPersonaLaunchFields, hasLaunchableCohort } from "./personaLaunchFields";
import { useCockpitBatchJob } from "./useCockpitBatchJob";
import { useSetupPersonaSampling } from "./useSetupPersonaSampling";

type HarborLaunchBody = Parameters<typeof api.launchHarborJob>[0];

/**
 * Cockpit-specific launch extras. Everything a cockpit may add on top of the
 * shared launch assembly; typed straight off the API client so the two can
 * never drift.
 */
export type CockpitLaunchOverrides = Pick<
  HarborLaunchBody,
  | "mode"
  | "agentName"
  | "cliSubscription"
  | "reasoningEffort"
  | "chatDomain"
  | "chatApplicationId"
  | "chatApplicationContext"
  | "chatMaxTurns"
  | "osAppSubmissionProfile"
  | "osAppBackend"
>;

/**
 * Shared launch assembly for the cockpits (#47).
 *
 * Owns everything every cockpit repeated by hand: the sampling + batch hook
 * pair, the launch-error state, the launchable-cohort guard, and the batch
 * launch call (persona fields, common body). A batch stays attached to the
 * workspace (center cells) until the operator confirms Config another run,
 * which backgrounds it on Runs and resets the cockpit. Cockpit-specific
 * launch fields go through `overrides`; debrief/live mappers and single-run
 * wiring stay in the cockpits, which own their result shapes.
 */
export function useCockpitLaunch(
  options: ConfigOptionsResponse | null,
  taskKind: HarborCockpitTaskKind,
  taskPath: string | null = null,
  isActive = true,
) {
  const sampling = useSetupPersonaSampling(options, taskKind, taskPath, isActive);
  const batch = useCockpitBatchJob(
    sampling.selectedPersonaIds,
    sampling.parallelTrials,
    taskKind,
    sampling.selectedCount,
    sampling.personaPool,
  );
  const queryClient = useQueryClient();
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [configAnotherOpen, setConfigAnotherOpen] = useState(false);
  const [batchLaunching, setBatchLaunching] = useState(false);

  const canLaunchCohort = hasLaunchableCohort({
    selectedPersonaIds: sampling.selectedPersonaIds,
    selectedCount: sampling.selectedCount,
    useEntirePool: sampling.useEntirePool,
  });

  const {
    personaPool,
    selectedPersonaIds,
    selectedCount,
    useEntirePool,
    parallelTrials,
    seed,
    personaModel,
  } = sampling;
  const { setBatchJobName, clearBatch, batchJobName } = batch;

  /**
   * Launch a batch job for `taskPath` and attach it to the workspace so the
   * center cells update. Resolves to true on success; on failure the message
   * lands in `launchError` and the call resolves to false.
   */
  const launchBatch = useCallback(
    async (input: {
      taskPath: string;
      taskId: string;
      overrides?: CockpitLaunchOverrides;
    }): Promise<boolean> => {
      setLaunchError(null);
      setBatchLaunching(true);
      try {
        const personaFields = buildPersonaLaunchFields({
          personaPool,
          selectedPersonaIds,
          selectedCount,
          useEntirePool,
          parallelTrials,
        });
        const launched = await api.launchHarborJob({
          taskPath: input.taskPath,
          seed,
          personaModel,
          ...personaFields,
          mode: "auto",
          ...input.overrides,
        });
        setBatchJobName(launched.jobName, {
          taskId: input.taskId,
          personaPool: personaPool ?? undefined,
        });
        void queryClient.invalidateQueries({ queryKey: ["harbor-jobs"] });
        return true;
      } catch (exc) {
        const message =
          exc instanceof ApiError ? exc.message : exc instanceof Error ? exc.message : String(exc);
        setLaunchError(message);
        return false;
      } finally {
        setBatchLaunching(false);
      }
    },
    [
      personaPool,
      selectedPersonaIds,
      selectedCount,
      useEntirePool,
      parallelTrials,
      seed,
      personaModel,
      queryClient,
      setBatchJobName,
    ],
  );

  /**
   * After the operator confirms: keep the batch running on Runs, unlock the
   * workspace, and reset persona setup so the next run is a clean config.
   */
  const confirmConfigAnotherRun = useCallback(() => {
    if (batchJobName) {
      void queryClient.invalidateQueries({ queryKey: ["harbor-jobs"] });
    }
    clearBatch();
    sampling.resetWorkspaceSetup();
    setLaunchError(null);
    setConfigAnotherOpen(false);
  }, [batchJobName, clearBatch, queryClient, sampling.resetWorkspaceSetup]);

  const requestConfigAnotherRun = useCallback(() => setConfigAnotherOpen(true), []);
  const cancelConfigAnotherRun = useCallback(() => setConfigAnotherOpen(false), []);
  const clearLaunchError = useCallback(() => setLaunchError(null), []);

  return {
    sampling,
    batch,
    launchError,
    setLaunchError,
    clearLaunchError,
    canLaunchCohort,
    launchBatch,
    requestConfigAnotherRun,
    confirmConfigAnotherRun,
    cancelConfigAnotherRun,
    configAnotherOpen,
    batchLaunching,
  };
}
