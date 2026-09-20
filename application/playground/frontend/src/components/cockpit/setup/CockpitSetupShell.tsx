import type { ReactNode } from "react";

export interface CockpitSetupShellProps {
  header: ReactNode;
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
}

/**
 * Cockpit setup shell.
 *
 * Wide (>= xl): full-viewport three-column cockpit — side rails scroll
 * internally, the page never scrolls.
 *
 * Narrow (< xl): the three rails stack into one scrolling column at their
 * natural height. Forcing the three-column geometry into a viewport that can't
 * hold it gave every rail a third of the height, which crammed their contents.
 * The stacked column is width-capped so it stays readable on mid-size windows.
 */
export function CockpitSetupShell({ header, left, center, right }: CockpitSetupShellProps) {
  return (
    <div className="cockpit-mesh-bg flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <div className="shrink-0 border-b border-outline/30 bg-surface-lowest/70 px-4 py-2 backdrop-blur-md sm:px-5">
        {header}
      </div>
      <div className="custom-scrollbar flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overflow-x-hidden px-3 py-2.5 xl:grid xl:grid-cols-12 xl:gap-4 xl:overflow-hidden xl:px-5 xl:py-3">
        {/* shrink-0 below xl: the rails keep their natural height and the column
            scrolls, instead of each section being squeezed into a third. */}
        <div className="mx-auto flex w-full max-w-5xl shrink-0 flex-col xl:col-span-3 xl:mx-0 xl:h-full xl:min-h-0 xl:shrink xl:overflow-hidden">
          {left}
        </div>
        <div className="mx-auto flex w-full max-w-5xl shrink-0 flex-col xl:col-span-6 xl:mx-0 xl:h-full xl:min-h-0 xl:shrink xl:overflow-hidden">
          {center}
        </div>
        <div className="mx-auto flex w-full max-w-5xl shrink-0 flex-col xl:col-span-3 xl:mx-0 xl:h-full xl:min-h-0 xl:shrink xl:overflow-hidden">
          {right}
        </div>
      </div>
    </div>
  );
}
