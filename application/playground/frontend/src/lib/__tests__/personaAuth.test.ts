import { describe, expect, it } from "vitest";

import {
  codexReasoningEfforts,
  formatUsdPer1M,
  personaAuthDefaultModel,
  personaAuthLaunchFields,
  personaAuthModelOptions,
} from "../personaAgentCatalog";

const MODEL_OPTIONS = [
  { value: "anthropic/claude-haiku-4-5", label: "Claude Haiku 4.5" },
  { value: "anthropic/claude-sonnet-5", label: "Claude Sonnet 5" },
  { value: "openai/gpt-4o-mini", label: "GPT-4o mini" },
  { value: "openai/gpt-5.4", label: "GPT-5.4" },
  { value: "openai/gpt-5.5", label: "GPT-5.5" },
  { value: "openai/gpt-6-sol", label: "GPT-6 Sol" },
  { value: "openai/gpt-6-luna", label: "GPT-6 Luna" },
  { value: "google/gemini-2.5-pro", label: "Gemini 2.5 Pro" },
];

describe("persona auth", () => {
  it("keeps the default API-key launch untouched", () => {
    expect(personaAuthLaunchFields("api", "xhigh")).toEqual({});
    expect(personaAuthModelOptions("api", MODEL_OPTIONS)).toEqual(MODEL_OPTIONS);
  });

  it("runs the Claude Code subscription on persona-claude-code with Anthropic models", () => {
    expect(personaAuthLaunchFields("claude-code", "xhigh")).toEqual({
      mode: "force_docker",
      agentName: "persona-claude-code",
      cliSubscription: true,
    });
    expect(personaAuthModelOptions("claude-code", MODEL_OPTIONS).map((opt) => opt.value)).toEqual([
      "anthropic/claude-haiku-4-5",
      "anthropic/claude-sonnet-5",
    ]);
  });

  it("starts each subscription on its default model", () => {
    expect(personaAuthDefaultModel("api")).toBeNull();
    expect(personaAuthDefaultModel("claude-code")).toBe("anthropic/claude-sonnet-5");
    expect(personaAuthDefaultModel("codex")).toBe("openai/gpt-5.5");
  });

  it("runs the Codex subscription on persona-codex with ChatGPT-account models only", () => {
    expect(personaAuthLaunchFields("codex", "xhigh")).toEqual({
      mode: "force_docker",
      agentName: "persona-codex",
      cliSubscription: true,
      reasoningEffort: "xhigh",
    });
    expect(personaAuthModelOptions("codex", MODEL_OPTIONS).map((opt) => opt.value)).toEqual([
      "openai/gpt-5.5",
      "openai/gpt-6-sol",
      "openai/gpt-6-luna",
    ]);
  });

  it("offers each Codex model only the reasoning efforts it supports", () => {
    expect(codexReasoningEfforts("openai/gpt-5.5")).toEqual(["low", "medium", "high", "xhigh"]);
    expect(codexReasoningEfforts("openai/gpt-6-sol")).toEqual([
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
      "ultra",
    ]);
    expect(codexReasoningEfforts("openai/gpt-6-luna")).toEqual([
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]);
    expect(codexReasoningEfforts("anthropic/claude-sonnet-5")).toEqual([]);
  });

  it("formats per-1M-token prices as compact USD", () => {
    expect(formatUsdPer1M(2)).toBe("$2");
    expect(formatUsdPer1M(2.5)).toBe("$2.5");
    expect(formatUsdPer1M(0.15)).toBe("$0.15");
  });
});
