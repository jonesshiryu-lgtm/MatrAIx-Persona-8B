"""Provider-aware credential helpers for MatrAIx job preflight.

Mirrors the prefix rules in ``playground.model_client.build_json_client`` so the
job generator and CLI can tell users which API key a job needs — without
printing secret values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Alternate key names ``playground.model_client`` also reads for these providers.
_ANTHROPIC_FALLBACKS: tuple[str, ...] = ("CLAUDE_API_KEY",)
_GEMINI_FALLBACKS: tuple[str, ...] = ("GOOGLE_API_KEY",)


def _env_set(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


@dataclass(frozen=True)
class ProviderCredential:
    """Resolved provider + credential env var for a persona ``model_name``.

    ``fallback_env_vars`` lists alternate env vars the runtime client also
    accepts (e.g. ``GOOGLE_API_KEY`` for Gemini, ``CLAUDE_API_KEY`` for
    Anthropic), so preflight does not report a usable key as missing.
    """

    provider: str
    env_var: str
    model_name: str
    fallback_env_vars: tuple[str, ...] = ()

    @property
    def resolved_env_var(self) -> str | None:
        """First env var (primary, then fallbacks) that is set, if any."""
        for name in (self.env_var, *self.fallback_env_vars):
            if _env_set(name):
                return name
        return None

    @property
    def present(self) -> bool:
        return self.resolved_env_var is not None


def resolve_provider_credential(model_name: str) -> ProviderCredential:
    """Infer the credential env var for a Harbor/MatrAIx ``model_name`` string."""
    value = (model_name or "").strip() or "anthropic/claude-sonnet-4-6"
    lowered = value.lower()

    if lowered.startswith("openai/") or lowered.startswith("gpt-"):
        return ProviderCredential("OpenAI", "OPENAI_API_KEY", value)
    if lowered.startswith("dashscope/"):
        return ProviderCredential("DashScope", "DASHSCOPE_API_KEY", value)
    if lowered.startswith("gemini/") or lowered.startswith("google/"):
        return ProviderCredential(
            "Gemini", "GEMINI_API_KEY", value, fallback_env_vars=_GEMINI_FALLBACKS
        )
    if lowered.startswith("openrouter/"):
        return ProviderCredential("OpenRouter", "OPENROUTER_API_KEY", value)
    if lowered.startswith("xai/"):
        return ProviderCredential("xAI", "XAI_API_KEY", value)
    if lowered.startswith("deepseek/"):
        return ProviderCredential("DeepSeek", "DEEPSEEK_API_KEY", value)
    if lowered.startswith("zai/"):
        return ProviderCredential("Z.ai", "ZAI_API_KEY", value)
    if lowered.startswith("anthropic/"):
        return ProviderCredential(
            "Anthropic",
            "ANTHROPIC_API_KEY",
            value,
            fallback_env_vars=_ANTHROPIC_FALLBACKS,
        )
    if "/" not in value:
        # Bare Anthropic model ids (historical default for Survey).
        return ProviderCredential(
            "Anthropic",
            "ANTHROPIC_API_KEY",
            value,
            fallback_env_vars=_ANTHROPIC_FALLBACKS,
        )
    # Unknown prefix: fall back to Anthropic so existing docs stay coherent,
    # but label the provider from the prefix for the preflight printout.
    prefix = value.split("/", 1)[0]
    return ProviderCredential(
        prefix, "ANTHROPIC_API_KEY", value, fallback_env_vars=_ANTHROPIC_FALLBACKS
    )


def format_credential_preflight(
    model_name: str,
    *,
    extra_env_hints: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Human-readable preflight lines (no secret values).

    ``extra_env_hints`` is a list of ``(env_var, reason)`` for secondary keys
    (e.g. user-sim chatbot engine still needing ``OPENAI_API_KEY``).
    """
    cred = resolve_provider_credential(model_name)
    resolved = cred.resolved_env_var
    if resolved == cred.env_var:
        credential_line = f"Credential: {cred.env_var} present"
    elif resolved:
        credential_line = (
            f"Credential: {resolved} present (used in place of {cred.env_var})"
        )
    elif cred.fallback_env_vars:
        alternates = " or ".join(cred.fallback_env_vars)
        credential_line = (
            f"Credential: {cred.env_var} missing ({alternates} also accepted)"
        )
    else:
        credential_line = f"Credential: {cred.env_var} missing"
    lines = [
        f"Provider: {cred.provider}",
        f"Model: {cred.model_name}",
        credential_line,
        "Credential value: not inspected",
    ]
    for env_var, reason in extra_env_hints or []:
        if env_var == cred.env_var:
            continue
        extra_status = "present" if _env_set(env_var) else "missing"
        lines.append(f"Also required: {env_var} {extra_status} ({reason})")
    return lines


def export_hint_lines(model_name: str) -> list[str]:
    """Shell export stubs for the primary (and only) model credential."""
    cred = resolve_provider_credential(model_name)
    return [f"export {cred.env_var}=..."]
