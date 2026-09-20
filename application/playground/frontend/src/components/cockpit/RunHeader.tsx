/**
 * Compact cockpit header — title + subtitle on the left, task-type switch on the right.
 * (No "Persona Cockpit" breadcrumb banner.)
 */
import { TaskTypeSwitch, type PlaygroundTaskType } from "./TaskTypeSwitch";
import { useI18n } from "@/i18n/I18nProvider";

export interface RunHeaderProps {
  taskType: PlaygroundTaskType;
  onTaskTypeChange: (value: PlaygroundTaskType) => void;
}

type Translate = ReturnType<typeof useI18n>["t"];

function subtitle(t: Translate, taskType: PlaygroundTaskType): string {
  switch (taskType) {
    case "chatbot": return t("runHeader.subtitle.chatbot");
    case "survey": return t("runHeader.subtitle.survey");
    case "web": return t("runHeader.subtitle.web");
    case "os-app": return t("runHeader.subtitle.osApp");
  }
}

/**
 * Dense header: title · subtitle · app-type switch.
 *
 * Below `xl` the three parts stack (title / full-width subtitle / switch) and
 * the subtitle wraps instead of being truncated to a few words while the switch
 * is crushed against it. From `xl` it collapses back to one line.
 */
export function RunHeader({ taskType, onTaskTypeChange }: RunHeaderProps) {
  const { t } = useI18n();
  const taskSubtitle = subtitle(t, taskType);
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <h1 className="shrink-0 font-display text-[18px] font-bold leading-tight tracking-tight text-text-main">
        {t("runHeader.title")}
      </h1>
      <p
        className="min-w-0 grow basis-full text-[13.5px] leading-snug text-text-variant xl:basis-64 xl:truncate"
        title={taskSubtitle}
      >
        {taskSubtitle}
      </p>
      <TaskTypeSwitch
        value={taskType}
        onChange={onTaskTypeChange}
        showLabel={false}
        className="flex w-full shrink-0 justify-start xl:ml-auto xl:w-auto xl:justify-end"
      />
    </div>
  );
}

export default RunHeader;
