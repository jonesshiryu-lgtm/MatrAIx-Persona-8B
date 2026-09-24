"""Tests for :class:`backend.service.config.ConfigManager` options metadata."""

from __future__ import annotations

from backend.service import config as config_module
from backend.service.config import PERSONA_MODEL_OPTIONS


def test_options_list_openai_proxy_chat_models(config_manager, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setattr(config_module, "_PROXY_MODELS_CACHE", {})
    monkeypatch.setattr(
        config_module,
        "_fetch_proxy_model_ids",
        lambda base, key: ["qwen3:8b", "[MLX] qwen3.5-122b", "bge-m3:latest", "[MLX] whisper-large-v3-turbo"],
    )
    knobs = {k["key"]: k for k in config_manager.options()["knobs"]}
    proxy = {o["value"]: o for o in knobs["personaModel"]["options"] if o.get("group")}
    assert list(proxy) == ["openai/qwen3:8b", "openai/[MLX] qwen3.5-122b"]
    assert proxy["openai/qwen3:8b"]["label"] == "qwen3:8b"
    assert proxy["openai/qwen3:8b"]["group"] == "OpenAI-compatible"


def test_fetch_proxy_model_ids_sends_user_agent(monkeypatch):
    # Some proxies (e.g. behind Cloudflare) 403 the default Python-urllib agent.
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"data": [{"id": "qwen3:8b"}]}'

    def _fake_urlopen(request, timeout):
        seen["user_agent"] = request.get_header("User-agent")
        seen["authorization"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr(config_module.urllib.request, "urlopen", _fake_urlopen)
    assert config_module._fetch_proxy_model_ids("https://proxy.example/v1", "k") == ["qwen3:8b"]
    assert seen["user_agent"] and "urllib" not in seen["user_agent"].lower()
    assert seen["authorization"] == "Bearer k"


def test_options_skip_proxy_models_without_base_url(config_manager, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(config_module, "_PROXY_MODELS_CACHE", {})
    monkeypatch.setattr(
        config_module, "_fetch_proxy_model_ids", lambda base, key: ["qwen3:8b"]
    )
    knobs = {k["key"]: k for k in config_manager.options()["knobs"]}
    assert "openai/qwen3:8b" not in [o["value"] for o in knobs["personaModel"]["options"]]


def test_domain_allows_all_three(config_manager):
    # ALLOWED still carries every domain; validation accepts each.
    assert set(config_manager.ALLOWED["domain"]) == {"movie", "beauty_product", "game"}
    for d in ("movie", "beauty_product", "game"):
        config_manager.validate({"domain": d})  # must not raise


def test_application_id_allows_generic_chatbot_applications(config_manager):
    assert config_manager.ALLOWED["applicationId"] == [
        "meal_planning_nutrition",
        "finance_openbb",
        "acme_support_api",
        "acme_support_mcp",
    ]
    for application_id in (
        "meal_planning_nutrition",
        "finance_openbb",
        "acme_support_api",
        "acme_support_mcp",
    ):
        config_manager.validate({"applicationId": application_id})


def test_options_returns_enriched_knobs(config_manager):
    opts = config_manager.options()
    assert set(opts.keys()) == {"knobs", "defaults", "environment"}

    knobs = {k["key"]: k for k in opts["knobs"]}
    # Editable knobs only — rankerMode / resourceMode are environment facts.
    assert set(knobs.keys()) == {
        "applicationId",
        "engine",
        "personaModel",
        "domain",
        "botType",
    }

    for knob in opts["knobs"]:
        assert set(knob.keys()) >= {
            "key",
            "label",
            "description",
            "options",
            "rebuildsAgent",
        }
        assert knob["label"]  # non-empty human label
        assert isinstance(knob["rebuildsAgent"], bool)
        for option in knob["options"]:
            assert set(option.keys()) >= {"value", "label", "description"}
            assert option["label"]


def test_options_knob_values_match_allowed(config_manager):
    opts = config_manager.options()
    knobs = {k["key"]: k for k in opts["knobs"]}
    for key in ("applicationId", "engine", "domain", "botType"):
        values = [o["value"] for o in knobs[key]["options"]]
        assert values == config_manager.ALLOWED[key]
    assert [o["value"] for o in knobs["personaModel"]["options"]] == PERSONA_MODEL_OPTIONS
    assert "anthropic/claude-opus-4-8" in PERSONA_MODEL_OPTIONS
    assert "anthropic/claude-sonnet-5" in PERSONA_MODEL_OPTIONS
    assert "openai/gpt-6-sol" in PERSONA_MODEL_OPTIONS
    assert "openai/gpt-6-luna" in PERSONA_MODEL_OPTIONS
    prices = {o["value"]: o for o in knobs["personaModel"]["options"]}
    assert prices["anthropic/claude-sonnet-5"]["inputCostPer1M"] == 2
    assert prices["anthropic/claude-sonnet-5"]["outputCostPer1M"] == 10
    assert prices["openai/gpt-5.5"]["inputCostPer1M"] == 5
    assert "dashscope/qwen3.7-max" in PERSONA_MODEL_OPTIONS
    assert "dashscope/deepseek-v4-pro" in PERSONA_MODEL_OPTIONS
    assert "openrouter/z-ai/glm-4.7" in PERSONA_MODEL_OPTIONS
    assert "openrouter/anthropic/claude-haiku-4.5" in PERSONA_MODEL_OPTIONS
    assert "openai/gpt-5.4" in PERSONA_MODEL_OPTIONS
    assert "openai/gpt-5.5" in PERSONA_MODEL_OPTIONS
    assert "gemini/gemini-2.5-flash" in PERSONA_MODEL_OPTIONS
    assert "gemini/gemini-2.5-pro" in PERSONA_MODEL_OPTIONS
    assert "gemini/gemini-2.5-computer-use-preview-10-2025" in PERSONA_MODEL_OPTIONS
    assert "xai/grok-4.5" in PERSONA_MODEL_OPTIONS
    assert "xai/grok-3-mini" in PERSONA_MODEL_OPTIONS
    assert "deepseek/deepseek-v4-pro" in PERSONA_MODEL_OPTIONS
    assert "deepseek/deepseek-chat" in PERSONA_MODEL_OPTIONS
    assert "zai/glm-5" in PERSONA_MODEL_OPTIONS
    assert "zai/glm-4.7" in PERSONA_MODEL_OPTIONS


def test_preflight_recognizes_openrouter_credentials(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    body = client.get("/api/preflight").json()
    model = next(c for c in body["checks"] if c["name"] == "Model credentials")
    assert model["ok"] is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    body = client.get("/api/preflight").json()
    model = next(c for c in body["checks"] if c["name"] == "Model credentials")
    openrouter = next(c for c in body["checks"] if c["name"] == "OpenRouter")
    assert model["ok"] is True
    assert "OpenRouter" in model["detail"]
    assert openrouter["ok"] is True
    assert openrouter.get("optional") is True


def test_options_rebuilds_agent_flag(config_manager):
    knobs = {k["key"]: k for k in config_manager.options()["knobs"]}
    # Every editable knob feeds the bridge's agent cache key, so each one
    # rebuilds (re-warms) the agent when changed — including botType, which is
    # part of INTERECAGENT_BOT_TYPE in the agent cache key.
    assert knobs["applicationId"]["rebuildsAgent"] is True
    assert knobs["domain"]["rebuildsAgent"] is True
    assert knobs["botType"]["rebuildsAgent"] is True
    assert knobs["engine"]["rebuildsAgent"] is True
    assert knobs["personaModel"]["rebuildsAgent"] is False


def test_bottype_change_requires_rebuild(config_manager):
    # Changing only botType must invalidate the cached agent (cold start),
    # because the bridge folds INTERECAGENT_BOT_TYPE into its agent cache key.
    old = {"botType": "chat"}
    new = {"botType": "completion"}
    assert config_manager.cache_invalidating(old, new) is True
    # botType is enumerated among the cache-invalidating keys.
    assert "botType" in config_manager.CACHE_INVALIDATING_KEYS
    # Sanity: an unchanged config does not force a rebuild.
    assert config_manager.cache_invalidating(old, dict(old)) is False


def test_options_defaults_are_full_config(config_manager):
    defaults = config_manager.options()["defaults"]
    # Full config: every key, including the fixed ranker/resource modes.
    assert set(defaults.keys()) == {
        "applicationId",
        "engine",
        "rankerMode",
        "resourceMode",
        "domain",
        "botType",
    }
    assert defaults["applicationId"] == "meal_planning_nutrition"
    assert defaults["engine"] == "gpt-4o-mini"
    assert defaults["domain"] == "movie"


def test_options_environment_block(config_manager):
    env = config_manager.options()["environment"]
    assert set(env.keys()) == {
        "runtime",
        "personaAgent",
        "personaModel",
        "applicationApi",
        "scorer",
        "cache",
        "ranker",
        "resources",
        "agent",
        "promptOwnership",
        "executionPlane",
        "remoteRunnerConfigured",
        "computeFamily",
        "computeFamilies",
    }
    assert env["runtime"] == "In-process Harbor runner"
    assert env["personaAgent"] == "Playground simulated user"
    assert env["personaModel"] == "anthropic/claude-haiku-4-5"
    assert env["applicationApi"] == "direct application adapter"
    assert env["scorer"] == "Playground self-report scorer"
    assert env["cache"] == "local service and model caches"
    assert "application-specific" in env["ranker"]
    assert "adapter-specific" in env["resources"]
    assert "chatbot application adapter" in env["agent"]
    assert env["promptOwnership"] == {
        "personaSystemPrompt": "Persona prompt from Playground",
        "taskPrompt": "Application-provided chatbot simulation prompt",
    }
