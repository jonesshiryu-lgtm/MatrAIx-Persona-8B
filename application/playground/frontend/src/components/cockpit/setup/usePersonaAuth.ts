import { useCallback, useEffect, useMemo, useState } from "react";

import {
  codexReasoningEfforts,
  DEFAULT_REASONING_EFFORT,
  personaAuthDefaultModel,
  personaAuthLaunchFields,
  personaAuthModelOptions,
  type PersonaAuth,
} from "@/lib/personaAgentCatalog";

import type { CockpitSelectOption } from "./CockpitSelect";

/**
 * Persona auth picker state for the Survey/Chatbot cockpits. A CLI
 * subscription pins one provider, so the model list (and the selected model)
 * follows the chosen auth; Codex models also carry a reasoning effort.
 * `launchFields` spread into single and batch launches.
 */
export function usePersonaAuth(
  personaModel: string,
  setPersonaModel: (model: string) => void,
  personaModelOptions: CockpitSelectOption[],
) {
  const [personaAuth, setPersonaAuthState] = useState<PersonaAuth>("api");
  const [reasoningEffort, setReasoningEffort] = useState(DEFAULT_REASONING_EFFORT);
  const modelOptions = useMemo(
    () => personaAuthModelOptions(personaAuth, personaModelOptions),
    [personaAuth, personaModelOptions],
  );
  const reasoningEfforts = useMemo(
    () => (personaAuth === "codex" ? codexReasoningEfforts(personaModel) : []),
    [personaAuth, personaModel],
  );
  const launchFields = useMemo(
    () => personaAuthLaunchFields(personaAuth, reasoningEffort),
    [personaAuth, reasoningEffort],
  );

  const setPersonaAuth = useCallback(
    (auth: PersonaAuth) => {
      setPersonaAuthState(auth);
      const preferred = personaAuthDefaultModel(auth);
      if (preferred && personaModelOptions.some((opt) => opt.value === preferred)) {
        setPersonaModel(preferred);
      }
    },
    [personaModelOptions, setPersonaModel],
  );

  useEffect(() => {
    if (personaAuth === "api" || modelOptions.length === 0) return;
    if (!modelOptions.some((opt) => opt.value === personaModel)) {
      setPersonaModel(modelOptions[0].value);
    }
  }, [personaAuth, modelOptions, personaModel, setPersonaModel]);

  useEffect(() => {
    if (reasoningEfforts.length === 0 || reasoningEfforts.includes(reasoningEffort)) return;
    setReasoningEffort(
      reasoningEfforts.includes(DEFAULT_REASONING_EFFORT)
        ? DEFAULT_REASONING_EFFORT
        : reasoningEfforts[0],
    );
  }, [reasoningEfforts, reasoningEffort]);

  return {
    personaAuth,
    setPersonaAuth,
    modelOptions,
    launchFields,
    reasoningEffort,
    setReasoningEffort,
    reasoningEfforts,
  };
}
