from matraix.provider_credentials import (
    format_credential_preflight,
    resolve_provider_credential,
)


def test_resolve_openai_and_dashscope(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    openai = resolve_provider_credential("openai/gpt-4o-mini")
    assert openai.provider == "OpenAI"
    assert openai.env_var == "OPENAI_API_KEY"
    assert openai.present is False
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    dash = resolve_provider_credential("dashscope/deepseek-v3.2")
    assert dash.provider == "DashScope"
    assert dash.env_var == "DASHSCOPE_API_KEY"
    assert dash.present is True


def test_resolve_gemini(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cred = resolve_provider_credential("gemini/gemini-2.5-pro")
    assert cred.provider == "Gemini"
    assert cred.env_var == "GEMINI_API_KEY"
    assert cred.present is False
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    assert resolve_provider_credential("google/gemini-2.5-flash").present is True


def test_resolve_xai(monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    cred = resolve_provider_credential("xai/grok-4.5")
    assert cred.provider == "xAI"
    assert cred.env_var == "XAI_API_KEY"
    assert cred.present is False
    monkeypatch.setenv("XAI_API_KEY", "sk-test")
    assert resolve_provider_credential("xai/grok-3-mini").present is True


def test_resolve_deepseek_and_zai(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    deepseek = resolve_provider_credential("deepseek/deepseek-v4-pro")
    assert deepseek.provider == "DeepSeek"
    assert deepseek.env_var == "DEEPSEEK_API_KEY"
    assert deepseek.present is False
    zai = resolve_provider_credential("zai/glm-4.7")
    assert zai.provider == "Z.ai"
    assert zai.env_var == "ZAI_API_KEY"
    assert zai.present is False
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("ZAI_API_KEY", "sk-test")
    assert resolve_provider_credential("deepseek/deepseek-chat").present is True
    assert resolve_provider_credential("zai/glm-5").present is True


def test_resolve_anthropic_default_and_bare_model() -> None:
    assert resolve_provider_credential("anthropic/claude-sonnet-4-6").env_var == (
        "ANTHROPIC_API_KEY"
    )
    assert resolve_provider_credential("claude-sonnet-4-6").provider == "Anthropic"


def test_gemini_accepts_google_api_key_fallback(monkeypatch) -> None:
    # playground.model_client reads GEMINI_API_KEY or GOOGLE_API_KEY.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google-secret")
    cred = resolve_provider_credential("gemini/gemini-2.5-pro")
    assert cred.env_var == "GEMINI_API_KEY"
    assert cred.present is True
    assert cred.resolved_env_var == "GOOGLE_API_KEY"
    joined = "\n".join(format_credential_preflight("gemini/gemini-2.5-pro"))
    assert "GOOGLE_API_KEY present (used in place of GEMINI_API_KEY)" in joined
    assert "missing" not in joined
    assert "sk-google-secret" not in joined


def test_anthropic_accepts_claude_api_key_fallback(monkeypatch) -> None:
    # AnthropicJSONClient reads ANTHROPIC_API_KEY or CLAUDE_API_KEY.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-claude-secret")
    for model in ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"):
        cred = resolve_provider_credential(model)
        assert cred.present is True
        assert cred.resolved_env_var == "CLAUDE_API_KEY"
    joined = "\n".join(format_credential_preflight("anthropic/claude-sonnet-4-6"))
    assert "CLAUDE_API_KEY present (used in place of ANTHROPIC_API_KEY)" in joined
    assert "sk-claude-secret" not in joined


def test_preflight_missing_mentions_accepted_alternates(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    lines = format_credential_preflight("google/gemini-2.5-flash")
    assert "Credential: GEMINI_API_KEY missing (GOOGLE_API_KEY also accepted)" in lines
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    lines = format_credential_preflight("openai/gpt-4o-mini")
    assert "Credential: OPENAI_API_KEY missing" in lines


def test_format_preflight_never_prints_secret(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-appear")
    lines = format_credential_preflight("openai/gpt-4o-mini")
    joined = "\n".join(lines)
    assert "OPENAI_API_KEY present" in joined
    assert "sk-secret" not in joined
    assert "Credential value: not inspected" in joined
