import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { useI18n } from "@/i18n/I18nProvider";
import { api, ApiError } from "@/lib/api";
import {
  PERSONA_BENCH_POOL,
  PERSONA_CARD_PREVIEW_LIMIT,
  PERSONA_GENERATE_COUNT_MAX,
  PERSONA_PRODUCTION_1M_POOL,
  PERSONA_SAMPLE_SIZE_MAX_DEV,
  PERSONA_SAMPLE_SIZE_MAX_PRODUCTION,
  PERSONA_UI_ID_LIST_MAX,
  type PersonaPoolGenerateProgress,
  type PersonaPoolGenerateResult,
  type PersonaPoolPersonaCard,
  type OverlayDimension,
  type TaskPersonaStrategy,
} from "@/lib/types";
import {
  classifyPersonaPoolSampleError,
  poolSlugLabel,
  type PersonaPoolSampleError,
} from "@/lib/personaPoolCopy";
import { syntheticDisplayName } from "@/lib/personaDisplay";
import { useDimensionLabels } from "@/lib/dimensionLabels";
import type { PersonaAuth } from "@/lib/personaAgentCatalog";
import { FOCUS_RING, Sym, humanizeToken } from "../cockpitShared";
import { CockpitSelect, type CockpitSelectOption } from "./CockpitSelect";
import { CockpitToggle } from "./CockpitToggle";
import { BenchPersonaCard } from "./BenchPersonaCard";
import { BenchPersonaDetailPanel } from "./BenchPersonaDetailPanel";
import { CockpitRailHeader } from "./CockpitRailHeader";
import { PersonaFilterModal } from "./PersonaFilterModal";
import {
  activeFilterCount,
  collectCatalogDimIds,
  emptyPersonaDimensionFilters,
  filterAxisIds,
  filterSelectionCounts,
  filtersForSampleApi,
  readStrategySampling,
  type PersonaDimensionFilters,
  type PersonaSamplingMode,
  type StratifiedAllocation,
  type StrategySamplingView,
} from "./personaSamplingTypes";
import type { PlaygroundTaskType } from "../TaskTypeSwitch";

type Translate = ReturnType<typeof useI18n>["t"];

const GENERATE_HINT_KEYS = {
  perCell: "personaSetup.generateHint.perCell",
  total: "personaSetup.generateHint.total",
  random: "personaSetup.generateDescription",
} as const;

function slugifyDatasetName(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function tabLabel(t: Translate, tab: PersonaSamplingMode): string {
  switch (tab) {
    case "single":
      return t("personaSetup.tabs.quick");
    case "random":
      return t("personaSetup.tabs.random");
    case "stratified":
      return t("personaSetup.tabs.stratified");
    case "all":
      return t("personaSetup.tabs.all");
  }
}

function tabTitle(t: Translate, tab: PersonaSamplingMode): string {
  switch (tab) {
    case "single":
      return t("personaSetup.tabTitles.quick");
    case "random":
      return t("personaSetup.tabTitles.random");
    case "stratified":
      return t("personaSetup.tabTitles.stratified");
    case "all":
      return t("personaSetup.tabTitles.all");
  }
}

const TAB_ORDER: PersonaSamplingMode[] = [
  "single",
  "random",
  "stratified",
  "all",
];

function allocationLabel(
  t: Translate,
  allocation: StratifiedAllocation,
): string {
  switch (allocation) {
    case "perCell":
      return t("personaSetup.allocation.perCell");
    case "proportional":
      return t("personaSetup.allocation.proportional");
    case "equalTotal":
      return t("personaSetup.allocation.equalTotal");
  }
}

function allocationTitle(
  t: Translate,
  allocation: StratifiedAllocation,
): string {
  switch (allocation) {
    case "perCell":
      return t("personaSetup.allocationTitle.perCell");
    case "proportional":
      return t("personaSetup.allocationTitle.proportional");
    case "equalTotal":
      return t("personaSetup.allocationTitle.equalTotal");
  }
}

const ALLOCATION_ORDER: StratifiedAllocation[] = [
  "perCell",
  "proportional",
  "equalTotal",
];

/** Default showcase personas from matraix-persona-dev-sample (smoke + spread). */
const QUICK_PICK_PERSONA_IDS = [
  "0042",
  "0001",
  "0328",
  "0058",
  "0012",
  "0020",
  "0030",
  "0040",
];

function clampSampleSize(
  value: number,
  max = PERSONA_SAMPLE_SIZE_MAX_DEV,
): number {
  if (!Number.isFinite(value)) return 4;
  return Math.min(max, Math.max(2, Math.round(value)));
}

function isProduction1mCohortPool(pool: string): boolean {
  return pool.includes("/matraix-persona-1m/cohorts/");
}

/** ``persona/datasets/<source>/cohorts/cohort-…`` — source dataset stays in the path. */
function parentDatasetFromCohortPool(pool: string): string | null {
  const match = pool.match(/^(persona\/datasets\/[^/]+)\/cohorts\/cohort-/);
  return match?.[1] ?? null;
}

function isSampleCohortPool(pool: string): boolean {
  const parent = parentDatasetFromCohortPool(pool);
  return Boolean(parent) && parent !== PERSONA_PRODUCTION_1M_POOL;
}

/** Pulled launch cache (1M or local dataset) — can be saved as a reusable dataset. */
function isMaterializedCohortPool(pool: string): boolean {
  return isProduction1mCohortPool(pool) || isSampleCohortPool(pool);
}

function isGeneratedDevPool(pool: string): boolean {
  return /(?:^|\/)generated-persona-dev-/.test(pool);
}

/** Custom Random count — small enough to try without a 2k synth. */
const CUSTOM_GENERATE_COUNT_DEFAULT = 10;

function clampGenerateCount(value: number): number {
  if (!Number.isFinite(value)) return CUSTOM_GENERATE_COUNT_DEFAULT;
  return Math.min(PERSONA_GENERATE_COUNT_MAX, Math.max(1, Math.round(value)));
}

function PersonaFilterChips({
  filters,
  fields,
  showStratify,
  overlayLabels,
}: {
  filters: PersonaDimensionFilters;
  fields: string[];
  showStratify: boolean;
  overlayLabels?: Record<string, string>;
}) {
  const { t } = useI18n();
  const labels = useDimensionLabels();
  const filterCount = activeFilterCount(filters);
  if (filterCount === 0 && !(showStratify && fields.length > 0)) return null;
  const dimName = (dim: string) =>
    overlayLabels?.[dim] || labels.dimLabel(dim, humanizeToken(dim));
  return (
    <div className="flex flex-wrap gap-1 px-0.5">
      {filters.sources.map((source) => (
        <span
          key={`src:${source}`}
          className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] text-primary"
        >
          {source}
        </span>
      ))}
      {Object.entries(filters.dimensionFilters)
        .filter(([, values]) => values.length > 0)
        .map(([dim, values]) => (
          <span
            key={dim}
            className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] text-primary"
            title={values
              .map((value) => labels.valueLabel(dim, value))
              .join(", ")}
          >
            {dimName(dim)}
            <span className="text-primary/70"> · {values.length}</span>
          </span>
        ))}
      {showStratify
        ? fields.map((field) => (
            <span
              key={`st:${field}`}
              className="rounded-full border border-secondary/35 bg-secondary/10 px-2 py-0.5 text-[11px] text-secondary"
            >
              {t("personaSetup.filters.stratify")} · {dimName(field)}
            </span>
          ))
        : null}
    </div>
  );
}

/** Dataset dropdown source — cohorts are launch caches, not selectable sources. */
function datasetSourcePool(pool: string): string {
  if (pool === PERSONA_PRODUCTION_1M_POOL || isProduction1mCohortPool(pool)) {
    return PERSONA_PRODUCTION_1M_POOL;
  }
  const parent = parentDatasetFromCohortPool(pool);
  if (parent) return parent;
  if (isSampleCohortPool(pool)) {
    return PERSONA_BENCH_POOL;
  }
  return pool;
}

function clampPerCell(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(50, Math.max(1, Math.round(value)));
}

function strategyModeLabel(
  t: Translate,
  mode: string | null | undefined,
): string {
  if (mode === "stratified") return t("personaSetup.tabs.stratified");
  if (mode === "random") return t("personaSetup.tabTitles.random");
  if (mode === "single") return t("personaSetup.tabTitles.quick");
  if (mode === "all") return t("personaSetup.tabs.all");
  return t("personaSetup.strategy.custom");
}

function strategySampleTypeLabel(
  t: Translate,
  sampling: StrategySamplingView,
): string | null {
  if (sampling.mode === "all") return t("personaSetup.strategy.entirePool");
  if (sampling.mode === "single")
    return t("personaSetup.personaCount", { count: 1 });
  if (sampling.mode === "stratified" && sampling.allocation === "perCell") {
    const n = sampling.perCell ?? 1;
    return t("personaSetup.strategy.personasPerCell", { count: n });
  }
  if (sampling.sampleSize != null) {
    return t("personaSetup.strategy.totalSample", {
      count: sampling.sampleSize,
    });
  }
  if (sampling.perCell != null) {
    return t("personaSetup.strategy.personasPerCell", {
      count: sampling.perCell,
    });
  }
  return null;
}

const STRATEGY_CHIP =
  "rounded-full px-2 py-0.5 text-[11px] leading-5 transition-colors duration-150";
const STRATEGY_FILTER_CHIP = `${STRATEGY_CHIP} bg-primary/10 text-primary hover:bg-primary/15`;
const STRATEGY_STRATIFY_CHIP = `${STRATEGY_CHIP} border border-secondary/35 bg-secondary/10 text-secondary`;

function StrategySectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="text-[10px] font-medium uppercase tracking-[0.08em] text-text-dim">
      {children}
    </p>
  );
}

function StrategyKvRow({
  label,
  children,
  title,
}: {
  label: string;
  children: ReactNode;
  title?: string;
}) {
  return (
    <div className="grid grid-cols-[4.75rem_minmax(0,1fr)] items-start gap-x-2 gap-y-0.5">
      <dt className="pt-0.5 text-[11px] text-text-dim">{label}</dt>
      <dd
        className="min-w-0 text-[12px] leading-snug text-text-main"
        title={title}
      >
        {children}
      </dd>
    </div>
  );
}

/** Compact dim chip — click to peek selected values. */
function StrategyFilterValueChip({
  label,
  values,
  open,
  onToggle,
}: {
  label: string;
  values: string[];
  open: boolean;
  onToggle: () => void;
}) {
  const { t } = useI18n();
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) onToggle();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onToggle();
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onToggle]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={onToggle}
        className={`${STRATEGY_FILTER_CHIP} cursor-pointer ${FOCUS_RING} ${
          open ? "ring-1 ring-primary/40" : ""
        }`}
        title={values.join(", ")}
      >
        {label}
        <span className="text-primary/70"> · {values.length}</span>
      </button>
      {open ? (
        <div
          role="dialog"
          aria-label={t("personaSetup.strategy.selectedValues", { label })}
          className="absolute left-0 top-[calc(100%+0.35rem)] z-20 w-[min(16.5rem,calc(100vw-2rem))] rounded-lg border border-outline/50 bg-surface-low p-2.5 shadow-[0_12px_28px_-16px_rgb(0_0_0/0.55)]"
        >
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <p className="min-w-0 truncate text-[11px] font-medium text-text-main">
              {label}
            </p>
            <p className="shrink-0 text-[10px] text-text-dim">
              {t("personaSetup.strategy.selectedCount", {
                count: values.length,
              })}
            </p>
          </div>
          <div className="flex max-h-40 flex-wrap gap-1 overflow-y-auto">
            {values.map((value) => (
              <span
                key={value}
                className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] leading-5 text-primary"
              >
                {value}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function strategyCollapsedHeadline(
  t: Translate,
  sampling: StrategySamplingView,
): string {
  const mode = strategyModeLabel(t, sampling.mode);
  const sample = strategySampleTypeLabel(t, sampling);
  if (sampling.mode === "stratified" || sampling.fields.length > 0) {
    const alloc = allocationLabel(t, sampling.allocation);
    return sample ? `${mode} · ${alloc} · ${sample}` : `${mode} · ${alloc}`;
  }
  return sample ? `${mode} · ${sample}` : mode;
}

function TaskStrategySummary({
  strategy,
  expanded,
  onExpandedChange,
}: {
  strategy: TaskPersonaStrategy;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
}) {
  const { t } = useI18n();
  const labels = useDimensionLabels();
  const sampling = readStrategySampling(strategy);
  const dimEntries = Object.entries(strategy.dimensionFilters ?? {}).filter(
    ([, values]) => Array.isArray(values) && values.length > 0,
  );
  const sources = (strategy.sources ?? []).filter((value) => value.trim());
  const stratify = sampling.fields;
  const filterBits = dimEntries.length + (sources.length > 0 ? 1 : 0);
  const showAllocation = sampling.mode === "stratified" || stratify.length > 0;
  const sampleType = strategySampleTypeLabel(t, sampling);
  const allocLabel = allocationLabel(t, sampling.allocation);
  // Declared target shares reweight a proportional draw; surface them and adjust
  // the allocation description away from the default "by cell population".
  const portionEntries = Object.entries(strategy.sampling?.portions ?? {}).filter(
    ([, weights]) =>
      weights && typeof weights === "object" && Object.keys(weights).length > 0,
  );
  const hasShares = portionEntries.length > 0;
  const allocDescription =
    hasShares && sampling.allocation === "proportional"
      ? t("personaSetup.allocationTitle.proportionalShares")
      : allocationTitle(t, sampling.allocation);
  const [openFilterKey, setOpenFilterKey] = useState<string | null>(null);

  useEffect(() => {
    if (!expanded) setOpenFilterKey(null);
  }, [expanded]);

  return (
    <div className="mt-2 border-t border-outline/35 pt-1.5">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => onExpandedChange(!expanded)}
        className={`flex w-full items-start gap-1.5 rounded-md py-0.5 text-left transition-colors duration-150 hover:bg-primary/5 ${FOCUS_RING}`}
      >
        <div className="min-w-0 flex-1">
          <p className="text-[12px] font-medium leading-snug text-text-main">
            {strategyCollapsedHeadline(t, sampling)}
          </p>
          {!expanded ? (
            <p className="mt-0.5 truncate text-[11px] leading-snug text-text-dim">
              {filterBits > 0
                ? t("personaSetup.strategy.filterCount", { count: filterBits })
                : t("personaSetup.strategy.noFilters")}
              {stratify.length > 0
                ? ` · ${t("personaSetup.strategy.stratifyFields", {
                    fields: stratify
                      .map((field) => labels.dimLabel(field, humanizeToken(field)))
                      .join(" × "),
                  })}`
                : ""}
            </p>
          ) : null}
        </div>
        <Sym
          name={expanded ? "expand_less" : "expand_more"}
          size={16}
          className="mt-0.5 shrink-0 text-text-dim"
        />
      </button>

      {expanded ? (
        <div className="mt-1 divide-y divide-outline/30">
          <section className="space-y-1.5 py-2.5">
            <StrategySectionLabel>
              {t("personaSetup.strategy.mode")}
            </StrategySectionLabel>
            <p className="text-[13px] font-medium leading-snug text-text-main">
              {strategyModeLabel(t, sampling.mode)}
            </p>
          </section>

          <section className="space-y-1.5 py-2.5">
            <StrategySectionLabel>
              {t("personaSetup.strategy.filterHeading", { count: filterBits })}
            </StrategySectionLabel>
            {filterBits > 0 ? (
              <div className="flex flex-wrap gap-1">
                {sources.length > 0 ? (
                  <StrategyFilterValueChip
                    label={t("personaSetup.strategy.sources")}
                    values={sources}
                    open={openFilterKey === "sources"}
                    onToggle={() =>
                      setOpenFilterKey((key) =>
                        key === "sources" ? null : "sources",
                      )
                    }
                  />
                ) : null}
                {dimEntries.map(([dim, values]) => (
                  <StrategyFilterValueChip
                    key={dim}
                    label={labels.dimLabel(dim, humanizeToken(dim))}
                    values={values.map((value) => labels.valueLabel(dim, value))}
                    open={openFilterKey === dim}
                    onToggle={() =>
                      setOpenFilterKey((key) => (key === dim ? null : dim))
                    }
                  />
                ))}
              </div>
            ) : (
              <p className="text-[12px] leading-snug text-text-dim">
                {t("personaSetup.strategy.noFiltersFullPool")}
              </p>
            )}
          </section>

          <section className="space-y-2 py-2.5">
            <StrategySectionLabel>
              {t("personaSetup.strategy.sampling")}
            </StrategySectionLabel>
            <dl className="space-y-1.5">
              {showAllocation ? (
                <StrategyKvRow
                  label={t("personaSetup.strategy.allocation")}
                  title={allocDescription}
                >
                  <span className="font-medium">{allocLabel}</span>
                  <span className="mt-0.5 block text-[11px] leading-snug text-text-dim">
                    {allocDescription}
                  </span>
                </StrategyKvRow>
              ) : null}
              {hasShares ? (
                <StrategyKvRow label={t("personaSetup.strategy.targetMix")}>
                  <div className="flex flex-col gap-1">
                    {portionEntries.map(([dim, weights]) => {
                      const total =
                        Object.values(weights).reduce(
                          (sum, w) => sum + (w > 0 ? w : 0),
                          0,
                        ) || 1;
                      return (
                        <div key={dim} className="flex flex-wrap gap-1">
                          {Object.entries(weights).map(([value, weight]) => (
                            <span key={value} className={STRATEGY_STRATIFY_CHIP}>
                              {labels.valueLabel(dim, value)}
                              <span className="text-secondary/70">
                                {" · "}
                                {Math.round((weight / total) * 100)}%
                              </span>
                            </span>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </StrategyKvRow>
              ) : null}
              {sampleType ? (
                <StrategyKvRow label={t("personaSetup.strategy.sample")}>
                  <span className="font-medium">{sampleType}</span>
                </StrategyKvRow>
              ) : null}
              {stratify.length > 0 || sampling.mode === "stratified" ? (
                <StrategyKvRow label={t("personaSetup.filters.stratify")}>
                  {stratify.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {stratify.map((field) => (
                        <span key={field} className={STRATEGY_STRATIFY_CHIP}>
                          {labels.dimLabel(field, humanizeToken(field))}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-text-dim">
                      {t("personaSetup.strategy.noStratifyAxes")}
                    </span>
                  )}
                </StrategyKvRow>
              ) : null}
            </dl>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function fallbackQuickPickCards(): PersonaPoolPersonaCard[] {
  return QUICK_PICK_PERSONA_IDS.map((personaId) => ({
    personaId,
    name: syntheticDisplayName(personaId),
    source: "matraix-persona-dev-sample",
    dimensions: {},
  }));
}

function GenerateProgressBar({
  progress,
}: {
  progress: Pick<
    PersonaPoolGenerateProgress,
    "ratio" | "label" | "stage"
  > & { datasetLabel?: string };
}) {
  const pct = Math.max(0, Math.min(100, Math.round(progress.ratio * 100)));
  const title = progress.datasetLabel || progress.label;
  return (
    <div className="space-y-1" aria-live="polite">
      <div className="flex items-center justify-between gap-2 text-[11px] leading-snug">
        <span className="min-w-0 truncate text-text-variant" title={title}>
          {title}
        </span>
        <span className="shrink-0 font-mono text-text-dim">{pct}%</span>
      </div>
      {progress.datasetLabel && progress.label !== progress.datasetLabel ? (
        <p className="truncate text-[10px] leading-snug text-text-dim">
          {progress.label}
        </p>
      ) : null}
      <div
        className="h-1.5 overflow-hidden rounded-full bg-outline/35"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label={title}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-200 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

type GenerateTrack = {
  key: string;
  label: string;
  ratio: number;
  detail: string;
  status: "pending" | "active" | "done";
};

function GenerateProgressTracks({ tracks }: { tracks: GenerateTrack[] }) {
  if (tracks.length === 0) return null;
  return (
    <div className="space-y-2" aria-live="polite">
      {tracks.map((track) => (
        <GenerateProgressBar
          key={track.key}
          progress={{
            ratio:
              track.status === "done"
                ? 1
                : track.status === "pending"
                  ? 0
                  : track.ratio,
            label: track.detail || track.label,
            datasetLabel: track.label,
            stage: track.status === "done" ? "done" : "write",
          }}
        />
      ))}
    </div>
  );
}

/** Task-plan (contract) synthesis: strategy-driven, read-only except cohort N. */
function ContractSynthBlock({
  strategy,
  view,
  customDims,
  contractSize,
  onContractSize,
  contractSizeDraft,
  onContractSizeDraft,
  seed,
  onSeed,
  strategyOpen,
  onStrategyOpen,
  generating,
  disabled,
  onGenerate,
  tracks,
  progress,
  error,
}: {
  strategy: TaskPersonaStrategy;
  view: {
    fill: "random" | "perCell" | "proportional" | "equalTotal";
    derivedN: number;
    sizeEditable: boolean;
  } | null;
  customDims: { id: string; values: string[] }[];
  contractSize: number | null;
  onContractSize: (n: number | null) => void;
  contractSizeDraft: string | null;
  onContractSizeDraft: (v: string | null) => void;
  seed: number;
  onSeed: (n: number) => void;
  strategyOpen: boolean;
  onStrategyOpen: (v: boolean) => void;
  generating: boolean;
  disabled: boolean;
  onGenerate: () => void;
  tracks: GenerateTrack[];
  progress: PersonaPoolGenerateProgress | null;
  error: string | null;
}) {
  const { t } = useI18n();
  const labels = useDimensionLabels();
  const effectiveN = contractSize ?? view?.derivedN ?? 0;
  const sizeEditable = view?.sizeEditable ?? false;
  return (
    <div className="space-y-2">
      <div className="glass-tile rounded-lg px-2.5 py-2">
        <TaskStrategySummary
          strategy={strategy}
          expanded={strategyOpen}
          onExpandedChange={onStrategyOpen}
        />
      </div>
      {customDims.length > 0 ? (
        <div className="rounded-lg border border-primary/25 bg-primary/[0.05] px-2.5 py-2">
          <p className="text-[11px] font-medium leading-snug text-text-main">
            {t("personaSetup.synth.customDimsNote", { count: customDims.length })}
          </p>
          <div className="mt-1 flex flex-wrap gap-1">
            {customDims.map((d) => (
              <span
                key={d.id}
                className="rounded bg-surface/70 px-1.5 py-0.5 text-[10.5px] text-text-dim"
                title={d.values.join(", ")}
              >
                {labels.dimLabel(d.id, humanizeToken(d.id))}
                <span className="text-text-dim/70"> · {d.values.length}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
      <div className="flex items-end gap-2">
        <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
          <span className="text-[12px] text-text-dim">
            {t("personaSetup.strategy.sample")}
          </span>
          {sizeEditable ? (
            <input
              type="number"
              inputMode="numeric"
              min={1}
              max={PERSONA_GENERATE_COUNT_MAX}
              step={1}
              value={contractSizeDraft ?? effectiveN}
              disabled={disabled || generating}
              onFocus={() => onContractSizeDraft(String(effectiveN))}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "" || /^\d+$/.test(raw)) onContractSizeDraft(raw);
              }}
              onBlur={() => {
                const raw = contractSizeDraft;
                onContractSizeDraft(null);
                if (raw === "" || raw == null) return;
                onContractSize(clampGenerateCount(Number(raw)));
              }}
              className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
            />
          ) : (
            <span
              className="flex h-9 w-full items-center justify-center rounded-lg border border-outline/40 bg-surface/40 px-1.5 text-center font-mono text-[15px] text-text-main"
              title={t("personaSetup.synth.sizeFromCells")}
            >
              {effectiveN}
            </span>
          )}
        </label>
        <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
          <span className="text-[12px] text-text-dim">
            {t("personaSetup.seed")}
          </span>
          <input
            type="number"
            inputMode="numeric"
            step={1}
            value={seed}
            disabled={disabled || generating}
            onChange={(e) => onSeed(Number(e.target.value) || 0)}
            className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
          />
        </label>
        <button
          type="button"
          disabled={disabled || generating}
          onClick={onGenerate}
          className={`flex h-9 min-w-0 flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-primary text-[13px] font-medium text-on-primary hover:opacity-90 disabled:opacity-50 ${FOCUS_RING}`}
        >
          <Sym
            name={generating ? "autorenew" : "auto_awesome"}
            size={15}
            className={generating ? "animate-rb-spin" : undefined}
          />
          {generating
            ? t("personaSetup.generating")
            : t("personaSetup.generate")}
        </button>
      </div>
      {tracks.length > 0 ? (
        <GenerateProgressTracks tracks={tracks} />
      ) : generating && progress ? (
        <GenerateProgressBar progress={progress} />
      ) : (
        <p className="text-[11px] leading-snug text-text-dim">
          {t("personaSetup.synth.taskPlanHint")}
        </p>
      )}
      {error ? (
        <div className="rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2">
          <p className="whitespace-pre-wrap text-[12px] leading-snug text-danger">
            {error}
          </p>
        </div>
      ) : null}
    </div>
  );
}

export interface PersonaSamplingRailProps {
  personaModel: string;
  onPersonaModelChange: (model: string) => void;
  personaModelOptions: CockpitSelectOption[];
  /** Persona billing: provider API key or a host CLI subscription (Claude Code / Codex). */
  personaAuth?: PersonaAuth;
  onPersonaAuthChange?: (auth: PersonaAuth) => void;
  /** Codex reasoning effort; the picker shows only when `reasoningEffortOptions` is non-empty. */
  reasoningEffort?: string;
  reasoningEffortOptions?: string[];
  onReasoningEffortChange?: (effort: string) => void;
  mode: PersonaSamplingMode;
  onModeChange: (mode: PersonaSamplingMode) => void;
  selectedPersonaIds: string[];
  onSelectedPersonaIdsChange: (ids: string[]) => void;
  selectedCount?: number;
  onSelectedCountChange?: (count: number) => void;
  useEntirePool?: boolean;
  onUseEntirePoolChange?: (value: boolean) => void;
  sampleSize: number;
  onSampleSizeChange: (size: number) => void;
  /** Personas per stratify combination; null when allocation is proportional / equalTotal. */
  perCell: number | null;
  onSampleSizePerValueGroupChange: (size: number) => void;
  /** Stratified allocation mode (only meaningful when mode === "stratified"). */
  stratifiedAllocation: StratifiedAllocation;
  onStratifiedAllocationChange: (allocation: StratifiedAllocation) => void;
  seed: number;
  filters: PersonaDimensionFilters;
  onFiltersChange: (filters: PersonaDimensionFilters) => void;
  fields: string[];
  onFieldsChange: (fields: string[]) => void;
  taskType?: PlaygroundTaskType;
  /** Active task path — used for strategy pool coverage recovery hints. */
  taskPath?: string | null;
  hasTaskStrategy?: boolean;
  taskPersonaStrategy?: TaskPersonaStrategy | null;
  useTaskDefaultStrategy?: boolean;
  onUseTaskDefaultStrategyChange?: (useDefault: boolean) => void;
  /** Called when sampling resolves to a (possibly materialized) pool path. */
  onPersonaPoolChange?: (pool: string) => void;
  /** Active pool path for the footer label (defaults to matraix-persona-dev-sample). */
  personaPool?: string | null;
  disabled?: boolean;
}

export function PersonaSamplingRail({
  personaModel,
  onPersonaModelChange,
  personaModelOptions,
  personaAuth = "api",
  onPersonaAuthChange,
  reasoningEffort = "",
  reasoningEffortOptions = [],
  onReasoningEffortChange,
  mode,
  onModeChange,
  selectedPersonaIds,
  onSelectedPersonaIdsChange,
  selectedCount = 0,
  onSelectedCountChange,
  useEntirePool = false,
  onUseEntirePoolChange,
  sampleSize,
  onSampleSizeChange,
  perCell,
  onSampleSizePerValueGroupChange,
  stratifiedAllocation,
  onStratifiedAllocationChange,
  seed,
  filters,
  onFiltersChange,
  fields,
  onFieldsChange,
  taskType,
  taskPath = null,
  hasTaskStrategy = false,
  taskPersonaStrategy = null,
  useTaskDefaultStrategy = false,
  onUseTaskDefaultStrategyChange,
  onPersonaPoolChange,
  personaPool = null,
  disabled,
}: PersonaSamplingRailProps) {
  const { t } = useI18n();
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterTarget, setFilterTarget] = useState<"dataset" | "generation">(
    "dataset",
  );
  const [railSegment, setRailSegment] = useState<"generation" | "dataset">(
    "dataset",
  );
  const [detailPersona, setDetailPersona] =
    useState<PersonaPoolPersonaCard | null>(null);
  const [previewCards, setPreviewCards] = useState<PersonaPoolPersonaCard[]>(
    [],
  );
  const [pullError, setPullError] = useState<PersonaPoolSampleError | null>(null);
  const [pulling, setPulling] = useState(false);
  /** Local draft so users can clear/retype without clamp fighting every keystroke. */
  const [sampleSizeDraft, setSampleSizeDraft] = useState<string | null>(null);
  const [perCellDraft, setPerCellDraft] = useState<string | null>(null);
  const [saveNameDraft, setSaveNameDraft] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [savingDataset, setSavingDataset] = useState(false);
  const [saveDatasetError, setSaveDatasetError] = useState<string | null>(null);
  const [genMode, setGenMode] = useState<"random" | "perCell" | "total">(
    "random",
  );
  const [genCount, setGenCount] = useState(CUSTOM_GENERATE_COUNT_DEFAULT);
  const [genCountDraft, setGenCountDraft] = useState<string | null>(null);
  const [genSeed, setGenSeed] = useState(42);
  const [genPerCell, setGenPerCell] = useState(2);
  const [genSampleSize, setGenSampleSize] = useState(32);
  const [genMarginals, setGenMarginals] = useState<
    Record<string, Record<string, number>>
  >({});
  const [genOverlay, setGenOverlay] = useState<OverlayDimension[]>([]);
  // Synthesis path: "contract" fills from the task's persona_strategy.json;
  // "custom" is the free-form random/by-cell/by-share builder below.
  const [genSynthMode, setGenSynthMode] = useState<"contract" | "custom">(
    hasTaskStrategy ? "contract" : "custom",
  );
  const genSynthTouched = useRef(false);
  const [genContractSize, setGenContractSize] = useState<number | null>(null);
  const [genContractSizeDraft, setGenContractSizeDraft] = useState<string | null>(
    null,
  );
  const [genStrategyOpen, setGenStrategyOpen] = useState(false);
  const pullErrorRecoveryHint =
    pullError?.showRecoveryHint
      ? t("personaSetup.errors.poolCoverageHint", {
          canSynthesize: taskPath?.trim() ? "true" : "false",
        })
      : null;
  const canSynthesizeCoverageError =
    pullError?.code === "persona_pool_coverage" && Boolean(taskPath?.trim());
  const [genFilters, setGenFilters] = useState<PersonaDimensionFilters>(
    emptyPersonaDimensionFilters,
  );
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [generateProgress, setGenerateProgress] =
    useState<PersonaPoolGenerateProgress | null>(null);
  const [generateTracks, setGenerateTracks] = useState<GenerateTrack[]>([]);
  /** Task-default strategy detail — collapses after a successful pull / synthesize. */
  const [strategySummaryOpen, setStrategySummaryOpen] = useState(true);

  const queryClient = useQueryClient();
  const activePool = personaPool?.trim() || PERSONA_BENCH_POOL;
  const sourcePool = datasetSourcePool(activePool);
  const isBenchPool = sourcePool === PERSONA_BENCH_POOL;
  const isProduction1m = sourcePool === PERSONA_PRODUCTION_1M_POOL;
  const canSaveAsDataset =
    (isMaterializedCohortPool(activePool) || isGeneratedDevPool(activePool)) &&
    (selectedCount > 0 || selectedPersonaIds.length > 0);
  const cohortSize = Math.max(selectedCount, selectedPersonaIds.length);
  const sampleSizeMax = isProduction1m
    ? PERSONA_SAMPLE_SIZE_MAX_PRODUCTION
    : PERSONA_SAMPLE_SIZE_MAX_DEV;
  const strategyLocked = hasTaskStrategy && useTaskDefaultStrategy;
  const customSamplingUnlocked = !strategyLocked;
  const strategyView = taskPersonaStrategy
    ? readStrategySampling(taskPersonaStrategy)
    : null;
  const panelMode = strategyLocked && strategyView ? strategyView.mode : mode;
  const panelAllocation =
    strategyLocked && strategyView
      ? strategyView.allocation
      : stratifiedAllocation;
  const panelFields =
    strategyLocked && strategyView ? strategyView.fields : fields;

  const datasetsQuery = useQuery({
    queryKey: ["persona-pool-datasets"],
    queryFn: () => api.listPersonaDatasets(),
    staleTime: 60_000,
  });

  const catalogQuery = useQuery({
    queryKey: ["persona-pool-catalog", sourcePool],
    queryFn: () => api.getPersonaPoolCatalog(sourcePool),
    staleTime: 60_000,
  });

  const defaultCardsQuery = useQuery({
    queryKey: [
      "persona-pool-default-cards",
      sourcePool,
      isBenchPool ? QUICK_PICK_PERSONA_IDS.join(",") : "preview",
    ],
    queryFn: async () => {
      try {
        return await api.getPersonaPoolCards({
          pool: sourcePool,
          limit: QUICK_PICK_PERSONA_IDS.length,
          personaIds: isBenchPool ? QUICK_PICK_PERSONA_IDS : undefined,
        });
      } catch {
        return {
          pool: sourcePool,
          personas: isBenchPool ? fallbackQuickPickCards() : [],
        };
      }
    },
    staleTime: 60_000,
  });

  const previewPersonaIds = useMemo(
    () => selectedPersonaIds.slice(0, PERSONA_CARD_PREVIEW_LIMIT),
    [selectedPersonaIds],
  );

  const lockedCohortQuery = useQuery({
    queryKey: [
      "persona-pool-locked-cohort",
      personaPool ?? PERSONA_BENCH_POOL,
      previewPersonaIds.join(","),
    ],
    queryFn: () =>
      api.getPersonaPoolCards({
        pool: personaPool ?? undefined,
        personaIds: previewPersonaIds,
        limit: previewPersonaIds.length,
      }),
    enabled:
      Boolean(disabled) && mode !== "single" && previewPersonaIds.length > 0,
    staleTime: 300_000,
  });

  const quickPickCards = useMemo(() => {
    const fromApi = defaultCardsQuery.data?.personas ?? [];
    if (fromApi.length > 0) return fromApi;
    if (defaultCardsQuery.isError) return fallbackQuickPickCards();
    return [];
  }, [defaultCardsQuery.data?.personas, defaultCardsQuery.isError]);

  const displayCards = useMemo(() => {
    if (panelMode === "single") return quickPickCards;
    if (disabled && selectedPersonaIds.length > 0) {
      const locked = lockedCohortQuery.data?.personas ?? [];
      if (locked.length > 0) return locked;
      return selectedPersonaIds.map((personaId) => ({
        personaId,
        name: syntheticDisplayName(personaId),
        source: activePool.split("/").filter(Boolean).pop() || "matraix-persona-dev-sample",
        dimensions: {},
      }));
    }
    return previewCards;
  }, [
    quickPickCards,
    previewCards,
    panelMode,
    disabled,
    selectedPersonaIds,
    lockedCohortQuery.data?.personas,
    activePool,
  ]);

  useEffect(() => {
    if (mode === "single") setPreviewCards([]);
  }, [mode]);

  // New task / re-enable Task default → show strategy detail again.
  useEffect(() => {
    if (useTaskDefaultStrategy) setStrategySummaryOpen(true);
  }, [taskPath, useTaskDefaultStrategy]);

  // Turning Task default on starts a fresh strategy pull — drop stale preview.
  // Turning it off keeps Dataset + selection, so leave preview cards in place.
  useEffect(() => {
    if (useTaskDefaultStrategy) {
      setPreviewCards([]);
      setPullError(null);
      setGenerateError(null);
    }
  }, [useTaskDefaultStrategy]);

  const togglePersona = useCallback(
    (personaId: string) => {
      if (useEntirePool) {
        // Cohort-ref selection is not per-card editable.
        return;
      }
      // Launch pool must match the Dataset those cards were loaded from.
      const cardPool = panelMode === "single" ? sourcePool : activePool;
      onPersonaPoolChange?.(cardPool);
      if (mode === "single") {
        onSelectedPersonaIdsChange(
          selectedPersonaIds.includes(personaId) ? [] : [personaId],
        );
        onSelectedCountChange?.(selectedPersonaIds.includes(personaId) ? 0 : 1);
        return;
      }
      const next = selectedPersonaIds.includes(personaId)
        ? selectedPersonaIds.filter((id) => id !== personaId)
        : [...selectedPersonaIds, personaId];
      onSelectedPersonaIdsChange(next);
      onSelectedCountChange?.(next.length);
    },
    [
      activePool,
      mode,
      onPersonaPoolChange,
      onSelectedCountChange,
      onSelectedPersonaIdsChange,
      panelMode,
      selectedPersonaIds,
      sourcePool,
      useEntirePool,
    ],
  );

  const handleSelectAll = useCallback(async () => {
    setPulling(true);
    setPullError(null);
    try {
      const pool = activePool;
      const idsResult = await api.listPersonaPoolIds(pool);
      const count = idsResult.count || idsResult.personaIds.length;
      if (count === 0) {
        throw new ApiError(404, t("personaSetup.errors.noPersonaFiles"));
      }
      const previewIds = idsResult.personaIds.slice(
        0,
        PERSONA_CARD_PREVIEW_LIMIT,
      );
      const preview = await api.getPersonaPoolCards({
        pool,
        limit: Math.min(PERSONA_CARD_PREVIEW_LIMIT, count),
        personaIds: previewIds.length ? previewIds : undefined,
      });
      setPreviewCards(preview.personas);
      const truncated =
        Boolean(idsResult.idsTruncated) || count > PERSONA_UI_ID_LIST_MAX;
      onSelectedCountChange?.(count);
      onUseEntirePoolChange?.(truncated);
      onSelectedPersonaIdsChange(
        truncated
          ? preview.personas.map((p) => p.personaId)
          : idsResult.personaIds,
      );
      onPersonaPoolChange?.(pool);
    } catch (err) {
      const raw =
        err instanceof ApiError
          ? err.message
          : t("personaSetup.errors.loadFullDataset");
      setPullError(classifyPersonaPoolSampleError(raw));
    } finally {
      setPulling(false);
    }
  }, [
    activePool,
    onPersonaPoolChange,
    onSelectedCountChange,
    onSelectedPersonaIdsChange,
    onUseEntirePoolChange,
    t,
  ]);

  const handlePull = useCallback(async () => {
    if (panelMode === "all") {
      await handleSelectAll();
      if (strategyLocked) setStrategySummaryOpen(false);
      return;
    }
    if (panelMode === "stratified" && panelFields.length === 0) {
      setPullError(
        classifyPersonaPoolSampleError(
          t("personaSetup.errors.pickStratifyAxis"),
        ),
      );
      return;
    }
    setPulling(true);
    setPullError(null);
    try {
      const dimensionFilters = filtersForSampleApi(filters);
      const isPerCell =
        panelMode === "stratified" && panelAllocation === "perCell";
      const perCellQuota = isPerCell ? clampPerCell(perCell ?? 1) : undefined;
      const strategyPortions = taskPersonaStrategy?.sampling?.portions;
      const result = await api.samplePersonaPool({
        pool: sourcePool,
        sampleSize,
        seed,
        sources: filters.sources.length ? filters.sources : undefined,
        dimensionFilters,
        fields: panelMode === "stratified" ? panelFields : undefined,
        perCell: perCellQuota,
        allocation: panelMode === "stratified" ? panelAllocation : undefined,
        portions:
          strategyPortions && Object.keys(strategyPortions).length > 0
            ? strategyPortions
            : undefined,
        taskPath: taskPath?.trim() || undefined,
      });
      const cards = result.personas
        .slice(0, PERSONA_CARD_PREVIEW_LIMIT)
        .map((row) => ({
          personaId: row.personaId,
          name: row.name ?? `persona-${row.personaId}`,
          source: row.source,
          path: row.path,
          dimensions: row.dimensions ?? {},
        }));
      setPreviewCards(cards);
      const count =
        result.selectedCount ?? result.sampleSize ?? result.personaIds.length;
      const truncated =
        Boolean(result.idsTruncated) || count > PERSONA_UI_ID_LIST_MAX;
      onSelectedCountChange?.(count);
      onUseEntirePoolChange?.(truncated);
      onSelectedPersonaIdsChange(result.personaIds);
      if (result.pool) {
        // Launch needs the materialized YAML dir; Dataset UI still shows sourcePool.
        onPersonaPoolChange?.(result.pool);
      }
      if (strategyLocked) setStrategySummaryOpen(false);
    } catch (err) {
      const raw =
        err instanceof ApiError
          ? err.message
          : t("personaSetup.errors.pullCohort");
      setPullError(classifyPersonaPoolSampleError(raw));
    } finally {
      setPulling(false);
    }
  }, [
    filters,
    handleSelectAll,
    onPersonaPoolChange,
    onSelectedCountChange,
    onSelectedPersonaIdsChange,
    onUseEntirePoolChange,
    panelAllocation,
    panelFields,
    panelMode,
    sampleSize,
    perCell,
    seed,
    sourcePool,
    strategyLocked,
    taskPath,
    taskPersonaStrategy,
    t,
  ]);

  const applyGeneratedPool = useCallback(
    async (
      result: PersonaPoolGenerateResult,
      options?: {
        /**
         * Task-default Synthesize replaces a failed Pull — select the fill cohort.
         * Custom Generation only materializes a Dataset; operator still Pulls.
         */
        selectCohort?: boolean;
      },
    ) => {
      await queryClient.invalidateQueries({
        queryKey: ["persona-pool-datasets"],
      });
      await queryClient.invalidateQueries({
        queryKey: ["persona-pool-catalog"],
      });
      const count = result.count;
      const ids = result.personaIds ?? [];
      const selectCohort = options?.selectCohort === true;
      const truncated = selectCohort && count > PERSONA_UI_ID_LIST_MAX;
      const previewIds = ids.slice(0, PERSONA_CARD_PREVIEW_LIMIT);
      onPersonaPoolChange?.(result.pool);
      if (selectCohort) {
        onSelectedCountChange?.(count);
        onUseEntirePoolChange?.(truncated);
        onSelectedPersonaIdsChange(truncated ? previewIds : ids);
      } else {
        // Pool is ready as Dataset — do not treat the whole draw as the launch cohort.
        onSelectedCountChange?.(0);
        onUseEntirePoolChange?.(false);
        onSelectedPersonaIdsChange([]);
        onModeChange("random");
      }
      try {
        const preview = await api.getPersonaPoolCards({
          pool: result.pool,
          limit: Math.min(
            PERSONA_CARD_PREVIEW_LIMIT,
            Math.max(selectCohort ? previewIds.length : 8, 1),
          ),
          personaIds:
            selectCohort && previewIds.length ? previewIds : undefined,
        });
        setPreviewCards(preview.personas);
      } catch {
        setPreviewCards(
          (selectCohort ? previewIds : ids.slice(0, 8)).map((personaId) => ({
            personaId,
            name: syntheticDisplayName(personaId),
            source: "synthetic",
            dimensions: {},
          })),
        );
      }
      setPullError(null);
      setGenerateError(null);
      setRailSegment("dataset");
      if (useTaskDefaultStrategy) setStrategySummaryOpen(false);
    },
    [
      onModeChange,
      onPersonaPoolChange,
      onSelectedCountChange,
      onSelectedPersonaIdsChange,
      onUseEntirePoolChange,
      queryClient,
      useTaskDefaultStrategy,
    ],
  );

  const genAxes = useMemo(() => filterAxisIds(genFilters), [genFilters]);
  const genOverlayLabels = useMemo(
    () => Object.fromEntries(genOverlay.map((dim) => [dim.id, dim.label])),
    [genOverlay],
  );
  const resetCustomGeneration = useCallback(() => {
    setGenMode("random");
    setGenCount(CUSTOM_GENERATE_COUNT_DEFAULT);
    setGenCountDraft(null);
    setGenPerCell(2);
    setGenSampleSize(32);
    setGenMarginals({});
    setGenOverlay([]);
    setGenFilters(emptyPersonaDimensionFilters);
  }, []);

  useEffect(() => {
    setGenMarginals((prev) => {
      const next: Record<string, Record<string, number>> = {};
      for (const dim of genAxes) {
        const values = genFilters.dimensionFilters[dim] ?? [];
        next[dim] = {};
        for (const value of values) {
          next[dim][value] = prev[dim]?.[value] ?? 1;
        }
      }
      return next;
    });
  }, [genAxes, genFilters]);

  // New task → wipe Custom. Toggling Task plan / Custom keeps that draft.
  useEffect(() => {
    genSynthTouched.current = false;
    setGenContractSize(null);
    resetCustomGeneration();
  }, [taskPath, resetCustomGeneration]);

  // Keep the synth path defaulted to the task plan once a strategy loads, until
  // the operator explicitly picks a path.
  useEffect(() => {
    if (genSynthTouched.current) return;
    setGenSynthMode(hasTaskStrategy ? "contract" : "custom");
  }, [hasTaskStrategy]);

  const isContractSynth =
    genSynthMode === "contract" && hasTaskStrategy && !!taskPersonaStrategy;

  // Contract synth is driven entirely by persona_strategy.json; derive a
  // read-only preview of what will be generated (fill mode + cohort size).
  const contractView = useMemo(() => {
    if (!taskPersonaStrategy) return null;
    const s = readStrategySampling(taskPersonaStrategy);
    const fill: "random" | "perCell" | "proportional" | "equalTotal" =
      s.mode === "random"
        ? "random"
        : s.allocation === "perCell"
          ? "perCell"
          : s.allocation === "proportional"
            ? "proportional"
            : "equalTotal";
    const filters = taskPersonaStrategy.dimensionFilters ?? {};
    const fieldIds = s.fields.length > 0 ? s.fields : Object.keys(filters);
    const combos = fieldIds.reduce(
      (n, f) => n * Math.max(1, (filters[f] ?? []).length),
      1,
    );
    const perCellN = s.perCell ?? 1;
    const derivedN =
      fill === "perCell"
        ? Math.max(1, combos * Math.max(1, perCellN))
        : Math.max(1, s.sampleSize ?? 8);
    return { fill, derivedN, sizeEditable: fill !== "perCell" };
  }, [taskPersonaStrategy]);

  // Strategy dims that are not part of the Full-DAG schema — the backend stamps
  // them as study overlays; surface them here so the operator sees what's custom.
  const contractCustomDims = useMemo(() => {
    if (!taskPersonaStrategy || !catalogQuery.data) return [];
    const schemaIds = collectCatalogDimIds(catalogQuery.data);
    const filters = taskPersonaStrategy.dimensionFilters ?? {};
    const out: { id: string; values: string[] }[] = [];
    for (const [dim, values] of Object.entries(filters)) {
      if (!Array.isArray(values) || values.length === 0) continue;
      if (!schemaIds.has(dim)) out.push({ id: dim, values });
    }
    return out;
  }, [taskPersonaStrategy, catalogQuery.data]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setGenerateError(null);
    setGenerateProgress(null);
    try {
      setGenerateTracks([
        {
          key: "dataset",
          label: t("personaSetup.progress.independentDataset"),
          ratio: 0,
          detail: t("personaSetup.progress.waiting"),
          status: "pending",
        },
      ]);
      const patchTracks = (event: PersonaPoolGenerateProgress) => {
        setGenerateTracks((prev) => {
          if (prev.length === 0) return prev;
          const index =
            typeof event.datasetIndex === "number" ? event.datasetIndex : 0;
          return prev.map((track, trackIndex) => {
            if (trackIndex !== index) {
              if (trackIndex < index && track.status !== "done") {
                return { ...track, status: "done", ratio: 1 };
              }
              return track;
            }
            const done = event.stage === "done" || event.ratio >= 0.999;
            return {
              ...track,
              label: event.datasetLabel || track.label,
              ratio: done ? 1 : event.ratio,
              detail: event.label || track.detail,
              status: done ? "done" : "active",
            };
          });
        });
        setGenerateProgress(event);
      };

      if (isContractSynth) {
        const path = taskPath?.trim();
        if (!path) {
          throw new ApiError(422, t("personaSetup.errors.selectTask"));
        }
        const result = await api.generatePersonaPool(
          {
            taskPath: path,
            seed: genSeed,
            ...(genContractSize != null ? { sampleSize: genContractSize } : {}),
          },
          { onProgress: patchTracks },
        );
        await applyGeneratedPool(result, { selectCohort: true });
        setGenerateTracks((prev) =>
          prev.map((track) => ({ ...track, status: "done", ratio: 1 })),
        );
        return;
      }

      const hasOverlay = genOverlay.length > 0;
      const filtersSelected =
        filterSelectionCounts(genFilters).attributes > 0 ||
        genFilters.sources.length > 0;
      if (
        (genMode === "perCell" || genMode === "total") &&
        genAxes.length === 0 &&
        !filtersSelected
      ) {
        throw new ApiError(422, t("personaSetup.errors.pickGenerateFilters"));
      }

      const needsFilters = genMode === "perCell" || genMode === "total";
      const overlayFilterMap = Object.fromEntries(
        genOverlay.map((dim) => [
          dim.id,
          genFilters.dimensionFilters[dim.id] ?? dim.values,
        ]),
      );
      const dimensionFilters = {
        ...(filtersForSampleApi(genFilters) ?? {}),
        ...(filtersForSampleApi({
          sources: [],
          dimensionFilters: overlayFilterMap,
        }) ?? {}),
      };
      const result = await api.generatePersonaPool(
        {
          count: genMode === "random" ? genCount : undefined,
          seed: genSeed,
          dimensionFilters:
            Object.keys(dimensionFilters).length > 0
              ? dimensionFilters
              : undefined,
          fields: needsFilters ? genAxes : undefined,
          perCell:
            genMode === "perCell" ? clampPerCell(genPerCell) : undefined,
          allocation:
            genMode === "perCell"
              ? "perCell"
              : genMode === "total"
                ? "independentMarginal"
                : undefined,
          sampleSize: genMode === "total" ? genSampleSize : undefined,
          marginals: genMode === "total" ? genMarginals : undefined,
          overlayDimensions: hasOverlay ? genOverlay : undefined,
        },
        { onProgress: patchTracks },
      );
      await applyGeneratedPool(result, { selectCohort: false });
      setGenerateTracks((prev) =>
        prev.map((track) => ({
          ...track,
          status: "done",
          ratio: 1,
        })),
      );
    } catch (err) {
      setGenerateError(
        err instanceof ApiError
          ? err.message
          : t("personaSetup.errors.generatePool"),
      );
    } finally {
      setGenerating(false);
      setGenerateProgress(null);
    }
  }, [
    applyGeneratedPool,
    genAxes,
    genContractSize,
    genCount,
    genFilters,
    genMarginals,
    genMode,
    genOverlay,
    genPerCell,
    genSampleSize,
    genSeed,
    isContractSynth,
    t,
    taskPath,
  ]);

  const handleSynthesizeTask = useCallback(async () => {
    const path = taskPath?.trim();
    if (!path) {
      setPullError(
        classifyPersonaPoolSampleError(t("personaSetup.errors.selectTask")),
      );
      return;
    }
    setGenerating(true);
    setPullError(null);
    setGenerateProgress({
      type: "progress",
      stage: "prepare",
      ratio: 0.02,
      label: t("personaSetup.progress.startingSynthesis"),
    });
    try {
      const result = await api.generatePersonaPool(
        { taskPath: path },
        { onProgress: setGenerateProgress },
      );
      // Synthesize stands in for Pull when coverage fails under Task default.
      await applyGeneratedPool(result, { selectCohort: true });
    } catch (err) {
      setPullError(
        classifyPersonaPoolSampleError(
          err instanceof ApiError
            ? err.message
            : t("personaSetup.errors.synthesizeTask"),
        ),
      );
    } finally {
      setGenerating(false);
      setGenerateProgress(null);
    }
  }, [applyGeneratedPool, t, taskPath]);

  const datasetOptions = useMemo<CockpitSelectOption[]>(() => {
    const listed = datasetsQuery.data?.datasets ?? [];
    const options: CockpitSelectOption[] = listed.map((item) => {
      const unavailable =
        item.kind === "production" && item.available === false;
      return {
        value: item.pool,
        label: item.label,
        meta: unavailable
          ? t("personaSetup.dataset.downloadHint")
          : item.count > 0
            ? t("personaSetup.personaCount", { count: item.count })
            : undefined,
      };
    });
    if (!options.some((opt) => opt.value === sourcePool)) {
      const slug =
        sourcePool.split("/").filter(Boolean).pop() ||
        "matraix-persona-dev-sample";
      options.unshift({
        value: sourcePool,
        label: slug,
      });
    }
    if (options.length === 0) {
      return [
        { value: PERSONA_BENCH_POOL, label: "matraix-persona-dev-sample" },
      ];
    }
    return options;
  }, [datasetsQuery.data?.datasets, sourcePool, t]);

  const handleDatasetChange = useCallback(
    (pool: string) => {
      if (pool === sourcePool) return;
      onPersonaPoolChange?.(pool);
      onSelectedPersonaIdsChange([]);
      onSelectedCountChange?.(0);
      onUseEntirePoolChange?.(false);
      setPreviewCards([]);
      setPullError(null);
      setDetailPersona(null);
      setSaveOpen(false);
      setSaveDatasetError(null);
      const option = datasetsQuery.data?.datasets?.find(
        (item) => item.pool === pool,
      );
      // Saved cohorts are already curated — default to All (inverse of 1M).
      if (option?.kind === "saved") {
        onModeChange("all");
      } else if (pool === PERSONA_PRODUCTION_1M_POOL && mode === "all") {
        onModeChange("random");
      }
    },
    [
      datasetsQuery.data?.datasets,
      mode,
      onModeChange,
      onPersonaPoolChange,
      onSelectedCountChange,
      onSelectedPersonaIdsChange,
      onUseEntirePoolChange,
      sourcePool,
    ],
  );

  const handleSaveAsDataset = useCallback(async () => {
    const name = saveNameDraft.trim() || `cohort-${cohortSize}`;
    const slug = slugifyDatasetName(name);
    if (!slug) {
      setSaveDatasetError(t("personaSetup.errors.invalidDatasetName"));
      return;
    }
    setSavingDataset(true);
    setSaveDatasetError(null);
    try {
      const saved = await api.savePersonaDataset({
        sourcePool: activePool,
        name,
      });
      await queryClient.invalidateQueries({
        queryKey: ["persona-pool-datasets"],
      });
      onPersonaPoolChange?.(saved.pool);
      onModeChange("all");
      setSaveOpen(false);
      setSaveNameDraft("");
    } catch (err) {
      setSaveDatasetError(
        err instanceof ApiError
          ? err.message
          : t("personaSetup.errors.saveDataset"),
      );
    } finally {
      setSavingDataset(false);
    }
  }, [
    activePool,
    cohortSize,
    onModeChange,
    onPersonaPoolChange,
    queryClient,
    saveNameDraft,
    t,
  ]);

  const poolCount = catalogQuery.data?.count;
  const showModelSelector =
    taskType === "survey" ||
    taskType === "chatbot" ||
    taskType === "web" ||
    taskType === "os-app";
  const poolFooterLabel = isMaterializedCohortPool(activePool)
    ? t("personaSetup.cohortReady", { pool: poolSlugLabel(sourcePool) })
    : poolSlugLabel(sourcePool);

  return (
    <aside className="glass-panel glass-panel-rail relative flex h-full min-h-0 flex-col rounded-xl p-4">
      {detailPersona ? (
        <BenchPersonaDetailPanel
          coverRail
          embedded
          persona={detailPersona}
          pool={sourcePool}
          onClose={() => setDetailPersona(null)}
          className="min-h-0 flex-1"
        />
      ) : (
        <>
          <div className="shrink-0">
            <CockpitRailHeader label={t("personaSetup.title")} />

            <div className="mb-2 space-y-1.5">
              {showModelSelector && (
                <CockpitSelect
                  label={t("personaSetup.model")}
                  inlineLabel
                  labelClassName="w-[4.25rem]"
                  value={personaModel}
                  options={personaModelOptions}
                  disabled={disabled}
                  wideMenu
                  onChange={onPersonaModelChange}
                />
              )}
              {showModelSelector && onPersonaAuthChange && (
                <CockpitSelect
                  label={t("personaSetup.auth")}
                  inlineLabel
                  labelClassName="w-[4.25rem]"
                  value={personaAuth}
                  options={[
                    { value: "api", label: t("personaSetup.auth.api") },
                    {
                      value: "claude-code",
                      label: t("personaSetup.auth.claudeCode"),
                      summary: t("personaSetup.auth.claudeCodeHint"),
                    },
                    {
                      value: "codex",
                      label: t("personaSetup.auth.codex"),
                      summary: t("personaSetup.auth.codexHint"),
                    },
                  ]}
                  disabled={disabled}
                  wideMenu
                  onChange={(value) => onPersonaAuthChange(value as PersonaAuth)}
                />
              )}
              {showModelSelector && onReasoningEffortChange && reasoningEffortOptions.length > 0 && (
                <CockpitSelect
                  label={t("personaSetup.reasoningEffort")}
                  inlineLabel
                  labelClassName="w-[4.25rem]"
                  value={reasoningEffort}
                  options={reasoningEffortOptions.map((effort) => ({ value: effort, label: effort }))}
                  disabled={disabled}
                  onChange={onReasoningEffortChange}
                />
              )}
            </div>

            <div className="cockpit-segment cockpit-segment--grid mb-2 grid-cols-2">
              {(["generation", "dataset"] as const).map((segment) => (
                <button
                  key={segment}
                  type="button"
                  disabled={disabled}
                  onClick={() => setRailSegment(segment)}
                  className={`cockpit-segment__btn cockpit-segment__btn--compact w-full ${FOCUS_RING} ${
                    railSegment === segment
                      ? "cockpit-segment__btn--active"
                      : ""
                  }`}
                >
                  {segment === "generation"
                    ? t("personaSetup.generation")
                    : t("personaSetup.dataset")}
                </button>
              ))}
            </div>
          </div>

          {railSegment === "generation" ? (
            <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto pr-0.5">
              <div className="mb-2 space-y-1.5">
                {hasTaskStrategy && taskPersonaStrategy ? (
                  <div className="cockpit-segment cockpit-segment--grid grid-cols-2">
                    {(["contract", "custom"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        disabled={disabled || generating}
                        onClick={() => {
                          genSynthTouched.current = true;
                          setGenSynthMode(m);
                        }}
                        className={`cockpit-segment__btn cockpit-segment__btn--compact w-full ${FOCUS_RING} ${
                          genSynthMode === m
                            ? "cockpit-segment__btn--active"
                            : ""
                        }`}
                      >
                        {m === "contract"
                          ? t("personaSetup.synth.taskPlan")
                          : t("personaSetup.synth.custom")}
                      </button>
                    ))}
                  </div>
                ) : null}
                {isContractSynth ? (
                  <ContractSynthBlock
                    strategy={taskPersonaStrategy!}
                    view={contractView}
                    customDims={contractCustomDims}
                    contractSize={genContractSize}
                    onContractSize={setGenContractSize}
                    contractSizeDraft={genContractSizeDraft}
                    onContractSizeDraft={setGenContractSizeDraft}
                    seed={genSeed}
                    onSeed={setGenSeed}
                    strategyOpen={genStrategyOpen}
                    onStrategyOpen={setGenStrategyOpen}
                    generating={generating}
                    disabled={!!disabled}
                    onGenerate={() => void handleGenerate()}
                    tracks={generateTracks}
                    progress={generateProgress}
                    error={generateError}
                  />
                ) : (
                  <>
                <div className="cockpit-segment cockpit-segment--grid grid-cols-3">
                  {(["random", "perCell", "total"] as const).map((tab) => (
                    <button
                      key={tab}
                      type="button"
                      title={t(
                        GENERATE_HINT_KEYS[
                          tab === "perCell" || tab === "total" ? tab : "random"
                        ],
                      )}
                      disabled={disabled || generating}
                      onClick={() => setGenMode(tab)}
                      className={`cockpit-segment__btn cockpit-segment__btn--compact w-full ${FOCUS_RING} ${
                        genMode === tab ? "cockpit-segment__btn--active" : ""
                      }`}
                    >
                      {tab === "random"
                        ? t("personaSetup.tabs.random")
                        : tab === "perCell"
                          ? t("personaSetup.tabs.byCombo")
                          : t("personaSetup.tabs.byShare")}
                    </button>
                  ))}
                </div>
                <div className="space-y-1.5">
                    <button
                      type="button"
                      disabled={disabled || generating}
                      onClick={() => {
                        setFilterTarget("generation");
                        setFilterOpen(true);
                      }}
                      className={`glass-tile glass-tile--hover flex w-full cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-left transition ${FOCUS_RING}`}
                    >
                      <Sym
                        name="tune"
                        size={16}
                        className="shrink-0 text-primary"
                      />
                      <span className="min-w-0 flex-1 text-[13px] font-medium text-text-main">
                        {t("personaSetup.filters.mixTitle")}
                      </span>
                      <span className="flex min-w-0 shrink-0 flex-col items-end gap-0.5 text-[11px] text-text-dim">
                        {filterSelectionCounts(genFilters).attributes > 0 ||
                        genFilters.sources.length > 0 ? (
                          <span>
                            {t(
                              "personaSetup.filters.filterCount",
                              filterSelectionCounts(genFilters),
                            )}
                          </span>
                        ) : null}
                        {filterSelectionCounts(genFilters).attributes === 0 &&
                        genFilters.sources.length === 0 &&
                        genOverlay.length > 0 ? (
                          <span>
                            {t("personaSetup.filters.overlayCount", {
                              count: genOverlay.length,
                            })}
                          </span>
                        ) : null}
                      </span>
                      <Sym
                        name="chevron_right"
                        size={16}
                        className="shrink-0 text-text-dim"
                      />
                    </button>
                    {filterSelectionCounts(genFilters).attributes > 0 ||
                    genFilters.sources.length > 0 ||
                    genOverlay.length > 0 ? (
                      <div className="space-y-1 rounded-lg border border-outline/25 px-2.5 py-2">
                        {filterSelectionCounts(genFilters).attributes > 0 ||
                        genOverlay.length > 0 ? (
                          <PersonaFilterChips
                            filters={genFilters}
                            fields={genAxes}
                            showStratify={false}
                            overlayLabels={genOverlayLabels}
                          />
                        ) : (
                          <p className="text-[11px] text-text-dim">
                            {t("personaSetup.filters.noMix")}
                          </p>
                        )}
                      </div>
                    ) : null}
                  </div>
                <div className="flex items-end gap-2">
                  {genMode === "random" ? (
                    <>
                      <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                        <span className="text-[12px] text-text-dim">
                          {t("personaSetup.count")}
                        </span>
                        <input
                          type="number"
                          inputMode="numeric"
                          min={1}
                          max={PERSONA_GENERATE_COUNT_MAX}
                          step={1}
                          value={genCountDraft ?? genCount}
                          disabled={disabled || generating}
                          onFocus={() => setGenCountDraft(String(genCount))}
                          onChange={(e) => {
                            const raw = e.target.value;
                            if (raw === "" || /^\d+$/.test(raw))
                              setGenCountDraft(raw);
                          }}
                          onBlur={() => {
                            const raw = genCountDraft;
                            setGenCountDraft(null);
                            setGenCount(
                              clampGenerateCount(
                                raw === "" || raw == null
                                  ? genCount
                                  : Number(raw),
                              ),
                            );
                          }}
                          className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                        />
                      </label>
                      <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                        <span className="text-[12px] text-text-dim">
                          {t("personaSetup.seed")}
                        </span>
                        <input
                          type="number"
                          inputMode="numeric"
                          step={1}
                          value={genSeed}
                          disabled={disabled || generating}
                          onChange={(e) =>
                            setGenSeed(Number(e.target.value) || 0)
                          }
                          className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                        />
                      </label>
                    </>
                  ) : (
                    <>
                      <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                        <span className="text-[12px] text-text-dim">
                          {genMode === "perCell"
                            ? t("personaSetup.perCell")
                            : t("personaSetup.strategy.sample")}
                        </span>
                        <input
                          type="number"
                          inputMode="numeric"
                          min={genMode === "perCell" ? 1 : 2}
                          max={
                            genMode === "perCell"
                              ? 50
                              : PERSONA_GENERATE_COUNT_MAX
                          }
                          step={1}
                          value={
                            genMode === "perCell" ? genPerCell : genSampleSize
                          }
                          disabled={disabled || generating}
                          onChange={(e) => {
                            const next = Number(e.target.value);
                            if (genMode === "perCell") {
                              setGenPerCell(clampPerCell(next));
                            } else {
                              setGenSampleSize(clampGenerateCount(next));
                            }
                          }}
                          className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                        />
                      </label>
                      <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                        <span className="text-[12px] text-text-dim">
                          {t("personaSetup.seed")}
                        </span>
                        <input
                          type="number"
                          inputMode="numeric"
                          step={1}
                          value={genSeed}
                          disabled={disabled || generating}
                          onChange={(e) =>
                            setGenSeed(Number(e.target.value) || 0)
                          }
                          className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                        />
                      </label>
                    </>
                  )}
                  <button
                    type="button"
                    disabled={disabled || generating}
                    onClick={() => void handleGenerate()}
                    className={`flex h-9 min-w-0 flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-primary text-[13px] font-medium text-on-primary hover:opacity-90 disabled:opacity-50 ${FOCUS_RING}`}
                  >
                    <Sym
                      name={generating ? "autorenew" : "auto_awesome"}
                      size={15}
                      className={generating ? "animate-rb-spin" : undefined}
                    />
                    {generating
                      ? t("personaSetup.generating")
                      : t("personaSetup.generate")}
                  </button>
                </div>
                {generateTracks.length > 0 ? (
                  <GenerateProgressTracks tracks={generateTracks} />
                ) : generating && generateProgress ? (
                  <GenerateProgressBar progress={generateProgress} />
                ) : (
                  <p className="text-[11px] leading-snug text-text-dim">
                    {t(
                      GENERATE_HINT_KEYS[
                        genMode === "perCell" || genMode === "total"
                          ? genMode
                          : "random"
                      ],
                    )}
                  </p>
                )}
                {generateError ? (
                  <div className="rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2">
                    <p className="whitespace-pre-wrap text-[12px] leading-snug text-danger">
                      {generateError}
                    </p>
                  </div>
                ) : null}
                  </>
                )}
              </div>
              <p className="rounded-lg border border-dashed border-outline/40 p-4 text-center text-[13px] leading-snug text-text-dim">
                {t("personaSetup.generationEmpty")}
              </p>
            </div>
          ) : (
            <>
              <div className="shrink-0">
                <div className="mb-2">
                  <CockpitSelect
                    label={t("personaSetup.dataset")}
                    inlineLabel
                    labelClassName="w-[4.25rem]"
                    value={sourcePool}
                    options={datasetOptions}
                    disabled={disabled}
                    showSelectedMeta
                    wideMenu
                    wrapOptions
                    onChange={handleDatasetChange}
                  />
                </div>
                {hasTaskStrategy && taskPersonaStrategy ? (
                  <div
                    className="glass-tile mb-2 rounded-lg px-2.5 py-2"
                    title={
                      useTaskDefaultStrategy
                        ? t("personaSetup.followStrategy")
                        : t("personaSetup.editFilters")
                    }
                  >
                    <CockpitToggle
                      label={t("personaSetup.taskDefaultStrategy")}
                      checked={useTaskDefaultStrategy}
                      disabled={disabled}
                      onChange={(checked) =>
                        onUseTaskDefaultStrategyChange?.(checked)
                      }
                    />
                    {useTaskDefaultStrategy ? (
                      <TaskStrategySummary
                        strategy={taskPersonaStrategy}
                        expanded={strategySummaryOpen}
                        onExpandedChange={setStrategySummaryOpen}
                      />
                    ) : null}
                  </div>
                ) : null}

                {strategyLocked ? (
                  <div className="mb-2 space-y-1.5">
                    <button
                      type="button"
                      disabled={disabled || pulling}
                      onClick={() => void handlePull()}
                      className={`flex h-9 w-full items-center justify-center gap-1.5 rounded-lg bg-surface-high/90 text-[13px] font-medium text-text-main hover:bg-surface-high disabled:opacity-50 ${FOCUS_RING}`}
                    >
                      <Sym
                        name={pulling ? "autorenew" : "download"}
                        size={15}
                        className={
                          pulling
                            ? "animate-rb-spin text-primary"
                            : "text-primary"
                        }
                      />
                      {pulling
                        ? t("personaSetup.pulling")
                        : t("personaSetup.pullCohort")}
                    </button>
                    {pullError ? (
                      <div className="space-y-1.5 rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2">
                        <p className="whitespace-pre-wrap text-[12px] leading-snug text-danger">
                          {pullError.rawMessage}
                        </p>
                        {pullErrorRecoveryHint ? (
                          <p className="text-[12px] leading-snug text-danger">
                            {pullErrorRecoveryHint}
                          </p>
                        ) : null}
                        {canSynthesizeCoverageError ? (
                          <div className="space-y-1.5">
                            <button
                              type="button"
                              disabled={disabled || generating}
                              onClick={() => void handleSynthesizeTask()}
                              className={`flex h-8 w-full items-center justify-center gap-1.5 rounded-md bg-primary/90 text-[12px] font-medium text-white hover:bg-primary disabled:opacity-50 ${FOCUS_RING}`}
                            >
                              <Sym
                                name={generating ? "autorenew" : "auto_awesome"}
                                size={14}
                                className={generating ? "animate-rb-spin" : ""}
                              />
                              {generating
                                ? t("personaSetup.synthesizing")
                                : t("personaSetup.synthesize")}
                            </button>
                            {generating && generateProgress ? (
                              <GenerateProgressBar
                                progress={generateProgress}
                              />
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {!strategyLocked ? (
                  <div className="cockpit-segment cockpit-segment--grid mb-2 grid-cols-4">
                    {TAB_ORDER.map((tab) => {
                      const allBlocked = tab === "all" && isProduction1m;
                      return (
                        <button
                          key={tab}
                          type="button"
                          title={
                            allBlocked
                              ? t("personaSetup.allDisabled")
                              : tabTitle(t, tab)
                          }
                          disabled={disabled || allBlocked}
                          onClick={() => onModeChange(tab)}
                          className={`cockpit-segment__btn cockpit-segment__btn--compact w-full ${FOCUS_RING} ${
                            panelMode === tab
                              ? "cockpit-segment__btn--active"
                              : ""
                          }`}
                        >
                          {tabLabel(t, tab)}
                        </button>
                      );
                    })}
                  </div>
                ) : null}

                {!strategyLocked && panelMode === "stratified" ? (
                  <div className="cockpit-segment cockpit-segment--grid mb-2 grid-cols-3">
                    {ALLOCATION_ORDER.map((alloc) => (
                      <button
                        key={alloc}
                        type="button"
                        title={allocationTitle(t, alloc)}
                        disabled={disabled || strategyLocked}
                        onClick={() => onStratifiedAllocationChange(alloc)}
                        className={`cockpit-segment__btn cockpit-segment__btn--compact w-full ${FOCUS_RING} ${
                          panelAllocation === alloc
                            ? "cockpit-segment__btn--active"
                            : ""
                        }`}
                      >
                        {allocationLabel(t, alloc)}
                      </button>
                    ))}
                  </div>
                ) : null}

                {disabled && selectedPersonaIds.length > 0 ? (
                  <p className="glass-tile mb-2 rounded-lg px-2.5 py-1.5 text-[12px] leading-snug text-text-variant">
                    {t("personaSetup.cohortLocked")}
                  </p>
                ) : null}

                {!strategyLocked && panelMode === "all" && (
                  <div className="mb-2 space-y-1.5">
                    <p className="glass-tile rounded-lg px-2.5 py-1.5 text-[12px] leading-snug text-text-variant">
                      {typeof poolCount === "number"
                        ? t("personaSetup.fullCohortWithCount", {
                            count: poolCount,
                          })
                        : t("personaSetup.fullCohort")}
                    </p>
                    <button
                      type="button"
                      disabled={disabled || pulling}
                      onClick={() => void handleSelectAll()}
                      className={`flex h-9 w-full items-center justify-center gap-1.5 rounded-lg bg-surface-high/90 text-[13px] font-medium text-text-main hover:bg-surface-high disabled:opacity-50 ${FOCUS_RING}`}
                    >
                      <Sym
                        name={pulling ? "autorenew" : "done_all"}
                        size={15}
                        className={
                          pulling
                            ? "animate-rb-spin text-primary"
                            : "text-primary"
                        }
                      />
                      {pulling
                        ? t("personaSetup.loadingCohort")
                        : typeof poolCount === "number"
                          ? t("personaSetup.selectAllCount", {
                              count: poolCount,
                            })
                          : t("personaSetup.selectAll")}
                    </button>
                    {pullError ? (
                      <div className="space-y-1.5 rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2">
                        <p className="text-[12px] leading-snug text-danger">
                          {pullError.rawMessage}
                        </p>
                        {pullErrorRecoveryHint ? (
                          <p className="text-[12px] leading-snug text-danger">
                            {pullErrorRecoveryHint}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                )}

                {!strategyLocked &&
                panelMode !== "single" &&
                panelMode !== "all" ? (
                  <div className="mb-2 space-y-1.5">
                    <div className="space-y-1.5">
                      {customSamplingUnlocked ? (
                        <button
                          type="button"
                          disabled={disabled}
                          onClick={() => {
                            setFilterTarget("dataset");
                            setFilterOpen(true);
                          }}
                          className={`glass-tile glass-tile--hover flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition ${FOCUS_RING}`}
                        >
                          <Sym
                            name="tune"
                            size={16}
                            className="shrink-0 text-primary"
                          />
                          <span className="min-w-0 flex-1 text-[13px] font-medium text-text-main">
                            {t("personaSetup.filters.title")}
                          </span>
                          {filterSelectionCounts(filters).attributes > 0 ? (
                            <span className="shrink-0 text-[11px] text-text-dim">
                              {t(
                                "personaSetup.filters.filterCount",
                                filterSelectionCounts(filters),
                              )}
                            </span>
                          ) : null}
                          {panelMode === "stratified" &&
                          panelFields.length > 0 ? (
                            <span
                              className="rounded-full bg-secondary/15 px-1.5 font-mono text-[11px] text-secondary"
                              title={t(
                                "personaSetup.filters.stratifyAxisCount",
                                { count: panelFields.length },
                              )}
                            >
                              {panelFields.length}×
                            </span>
                          ) : null}
                          <Sym
                            name="chevron_right"
                            size={16}
                            className="shrink-0 text-text-dim"
                          />
                        </button>
                      ) : null}
                      {customSamplingUnlocked ? (
                        <PersonaFilterChips
                          filters={filters}
                          fields={panelFields}
                          showStratify={panelMode === "stratified"}
                        />
                      ) : null}
                    </div>

                    <div className="flex items-end gap-2">
                      {panelMode === "stratified" &&
                      panelAllocation === "perCell" ? (
                        <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                          <span className="text-[12px] text-text-dim">
                            {t("personaSetup.perCell")}
                          </span>
                          <input
                            type="number"
                            inputMode="numeric"
                            min={1}
                            max={50}
                            step={1}
                            value={perCellDraft ?? perCell ?? 1}
                            disabled={disabled || strategyLocked}
                            onFocus={() =>
                              setPerCellDraft(String(perCell ?? 1))
                            }
                            onChange={(e) => {
                              const raw = e.target.value;
                              if (raw === "" || /^\d+$/.test(raw)) {
                                setPerCellDraft(raw);
                              }
                            }}
                            onBlur={() => {
                              const raw = perCellDraft;
                              setPerCellDraft(null);
                              onSampleSizePerValueGroupChange(
                                clampPerCell(
                                  raw === "" || raw == null
                                    ? (perCell ?? 1)
                                    : Number(raw),
                                ),
                              );
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                (e.target as HTMLInputElement).blur();
                              }
                            }}
                            className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                          />
                        </label>
                      ) : (
                        <label className="flex w-[4.25rem] shrink-0 flex-col gap-0.5">
                          <span className="text-[12px] text-text-dim">
                            {t("personaSetup.strategy.sample")}
                          </span>
                          <input
                            type="number"
                            inputMode="numeric"
                            min={2}
                            max={sampleSizeMax}
                            step={1}
                            value={sampleSizeDraft ?? sampleSize}
                            disabled={disabled || strategyLocked}
                            onFocus={() =>
                              setSampleSizeDraft(String(sampleSize))
                            }
                            onChange={(e) => {
                              const raw = e.target.value;
                              if (raw === "" || /^\d+$/.test(raw)) {
                                setSampleSizeDraft(raw);
                              }
                            }}
                            onBlur={() => {
                              const raw = sampleSizeDraft;
                              setSampleSizeDraft(null);
                              onSampleSizeChange(
                                clampSampleSize(
                                  raw === "" || raw == null
                                    ? sampleSize
                                    : Number(raw),
                                  sampleSizeMax,
                                ),
                              );
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                (e.target as HTMLInputElement).blur();
                              }
                            }}
                            className={`h-9 w-full rounded-lg border border-outline/50 bg-surface/60 px-1.5 text-center font-mono text-[15px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                          />
                        </label>
                      )}
                      <button
                        type="button"
                        disabled={disabled || pulling}
                        onClick={() => void handlePull()}
                        className={`flex h-9 min-w-0 flex-1 items-center justify-center gap-1.5 rounded-lg bg-surface-high/90 text-[13px] font-medium text-text-main hover:bg-surface-high disabled:opacity-50 ${FOCUS_RING}`}
                      >
                        <Sym
                          name={pulling ? "autorenew" : "download"}
                          size={15}
                          className={
                            pulling
                              ? "animate-rb-spin text-primary"
                              : "text-primary"
                          }
                        />
                        {pulling
                          ? t("personaSetup.pulling")
                          : t("personaSetup.pullCohort")}
                      </button>
                    </div>
                    {pullError ? (
                      <div className="space-y-1.5 rounded-lg border border-danger/30 bg-danger/5 px-2.5 py-2">
                        <p className="whitespace-pre-wrap text-[12px] leading-snug text-danger">
                          {pullError.rawMessage}
                        </p>
                        {pullErrorRecoveryHint ? (
                          <p className="text-[12px] leading-snug text-danger">
                            {pullErrorRecoveryHint}
                          </p>
                        ) : null}
                        {useTaskDefaultStrategy && canSynthesizeCoverageError ? (
                          <div className="space-y-1.5">
                            <button
                              type="button"
                              disabled={disabled || generating}
                              onClick={() => void handleSynthesizeTask()}
                              className={`flex h-8 w-full items-center justify-center gap-1.5 rounded-md bg-primary/90 text-[12px] font-medium text-white hover:bg-primary disabled:opacity-50 ${FOCUS_RING}`}
                            >
                              <Sym
                                name={generating ? "autorenew" : "auto_awesome"}
                                size={14}
                                className={generating ? "animate-rb-spin" : ""}
                              />
                              {generating
                                ? t("personaSetup.synthesizing")
                                : t("personaSetup.synthesize")}
                            </button>
                            {generating && generateProgress ? (
                              <GenerateProgressBar
                                progress={generateProgress}
                              />
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    {canSaveAsDataset && !disabled ? (
                      <div className="rounded-lg border border-outline/40 bg-surface/40 px-2.5 py-2">
                        {!saveOpen ? (
                          <button
                            type="button"
                            disabled={savingDataset}
                            onClick={() => {
                              setSaveOpen(true);
                              setSaveDatasetError(null);
                              if (!saveNameDraft.trim()) {
                                const stamp = new Date()
                                  .toISOString()
                                  .slice(0, 10);
                                setSaveNameDraft(
                                  `sample-${cohortSize}-${stamp}`,
                                );
                              }
                            }}
                            className={`flex w-full items-center justify-center gap-1.5 rounded-md py-1.5 text-[12px] font-medium text-primary hover:bg-primary/5 disabled:opacity-50 ${FOCUS_RING}`}
                          >
                            <Sym name="save" size={14} />
                            {t("personaSetup.saveAsDataset")}
                          </button>
                        ) : (
                          <div className="space-y-1.5">
                            <label className="block space-y-1">
                              <span className="text-[11px] font-medium uppercase tracking-wide text-text-dim">
                                {t("personaSetup.datasetName")}
                              </span>
                              <input
                                type="text"
                                value={saveNameDraft}
                                disabled={savingDataset}
                                placeholder="my-robinhood-cohort"
                                onChange={(e) =>
                                  setSaveNameDraft(e.target.value)
                                }
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") {
                                    e.preventDefault();
                                    void handleSaveAsDataset();
                                  }
                                  if (e.key === "Escape") {
                                    setSaveOpen(false);
                                    setSaveDatasetError(null);
                                  }
                                }}
                                className={`h-8 w-full rounded-md border border-outline/50 bg-field px-2 font-mono text-[12px] text-text-main disabled:opacity-50 ${FOCUS_RING}`}
                              />
                            </label>
                            {saveNameDraft.trim() ? (
                              <p className="truncate font-mono text-[10px] text-text-dim">
                                persona/datasets/
                                {slugifyDatasetName(saveNameDraft) || "…"}
                              </p>
                            ) : null}
                            {saveDatasetError ? (
                              <p className="text-[11px] leading-snug text-danger">
                                {saveDatasetError}
                              </p>
                            ) : (
                              <p className="text-[11px] leading-snug text-text-dim">
                                {t("personaSetup.datasetReuse")}
                              </p>
                            )}
                            <div className="flex gap-1.5">
                              <button
                                type="button"
                                disabled={savingDataset}
                                onClick={() => {
                                  setSaveOpen(false);
                                  setSaveDatasetError(null);
                                }}
                                className={`h-8 flex-1 rounded-md border border-outline/50 text-[12px] text-text-variant hover:bg-surface-high/60 disabled:opacity-50 ${FOCUS_RING}`}
                              >
                                {t("personaSetup.common.cancel")}
                              </button>
                              <button
                                type="button"
                                disabled={
                                  savingDataset || !saveNameDraft.trim()
                                }
                                onClick={() => void handleSaveAsDataset()}
                                className={`h-8 flex-1 rounded-md bg-primary/90 text-[12px] font-medium text-white hover:bg-primary disabled:opacity-50 ${FOCUS_RING}`}
                              >
                                {savingDataset
                                  ? t("personaSetup.saving")
                                  : t("personaSetup.save")}
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
              <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto pr-0.5">
                <div className="space-y-2">
                  {panelMode === "single" &&
                    defaultCardsQuery.isLoading &&
                    quickPickCards.length === 0 && (
                      <p className="text-[13px] text-text-variant">
                        {t("personaSetup.loadingDataset", {
                          dataset:
                            activePool.split("/").filter(Boolean).pop() ||
                            t("personaSetup.datasetFallback"),
                        })}
                      </p>
                    )}
                  {panelMode === "single" &&
                    defaultCardsQuery.isError &&
                    quickPickCards.length > 0 && (
                      <p className="text-[12px] text-warn">
                        {t("personaSetup.offlineList")}
                      </p>
                    )}
                  {displayCards.map((persona) => (
                    <BenchPersonaCard
                      key={persona.personaId}
                      persona={persona}
                      selected={selectedPersonaIds.includes(persona.personaId)}
                      disabled={disabled}
                      onToggle={() => togglePersona(persona.personaId)}
                      onOpenDetail={() => setDetailPersona(persona)}
                    />
                  ))}
                  {panelMode !== "single" &&
                    cohortSize > displayCards.length &&
                    displayCards.length > 0 && (
                      <p className="rounded-lg border border-dashed border-outline/40 px-3 py-2 text-center text-[12px] text-text-dim">
                        {t("personaSetup.previewing", {
                          shown: displayCards.length,
                          selected: cohortSize,
                        })}
                        {useEntirePool
                          ? t("personaSetup.previewingCohortRef")
                          : t("personaSetup.previewingFullCohort")}
                      </p>
                    )}
                  {panelMode === "single" &&
                    !defaultCardsQuery.isLoading &&
                    displayCards.length === 0 && (
                      <p className="rounded-lg border border-dashed border-outline/40 p-4 text-center text-[13px] text-text-dim">
                        {t("personaSetup.noneLoaded")}
                      </p>
                    )}
                  {panelMode !== "single" &&
                    displayCards.length === 0 &&
                    !disabled && (
                      <p className="rounded-lg border border-dashed border-outline/40 p-4 text-center text-[13px] text-text-dim">
                        {strategyLocked
                          ? t("personaSetup.emptyStrategy")
                          : panelMode === "all"
                            ? t("personaSetup.emptyAll")
                            : t("personaSetup.emptyCustom")}
                      </p>
                    )}
                  {panelMode !== "single" &&
                    displayCards.length === 0 &&
                    disabled &&
                    lockedCohortQuery.isLoading && (
                      <p className="text-[13px] text-text-variant">
                        {t("personaSetup.loadingCohortPersonas")}
                      </p>
                    )}
                </div>
              </div>

              <p className="mt-2 shrink-0 truncate text-center font-mono text-[12px] tracking-wide text-text-dim">
                {t("personaSetup.selectedCount", { count: cohortSize })} ·{" "}
                {poolFooterLabel}
              </p>
            </>
          )}
        </>
      )}

      <PersonaFilterModal
        open={filterOpen}
        catalog={catalogQuery.data ?? null}
        filters={filterTarget === "generation" ? genFilters : filters}
        stratifyMode={
          filterTarget === "generation" ? false : panelMode === "stratified"
        }
        fields={filterTarget === "generation" ? [] : fields}
        onFieldsChange={
          filterTarget === "generation" ? undefined : onFieldsChange
        }
        showMarginals={filterTarget === "generation" && genMode === "total"}
        allowOverlayEdit={filterTarget === "generation"}
        overlayDimensions={
          filterTarget === "generation" ? genOverlay : undefined
        }
        marginals={
          filterTarget === "generation" && genMode === "total"
            ? genMarginals
            : undefined
        }
        personaModel={personaModel}
        onClose={() => setFilterOpen(false)}
        onConfirm={(next, nextMarginals, nextOverlay) => {
          if (filterTarget === "generation") {
            if (nextOverlay !== undefined) setGenOverlay(nextOverlay);
            setGenFilters(next);
            if (nextMarginals) setGenMarginals(nextMarginals);
            return;
          }
          onFiltersChange(next);
        }}
      />
    </aside>
  );
}
