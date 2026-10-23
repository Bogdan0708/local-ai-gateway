"""Agents use the configured chat backend (e.g. LM Studio), not a LiteLLM proxy or its aliases."""

import pytest

from src.agents import DevOpsAgent, ResearchAgent, create_agent


def models(agent):
    if isinstance(agent, DevOpsAgent):
        return {"fast": agent.fast_model, "quality": agent.quality_model}
    return {"ultra": agent.ultra_model}


@pytest.mark.parametrize("agent_cls", [DevOpsAgent, ResearchAgent])
def test_agents_target_the_active_backend_and_chat_model(settings_env, agent_cls):
    settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_HOST="http://studio.invalid:1234/",
                 LMSTUDIO_CHAT_MODEL="hub-model", OPENAI_API_KEY="synthetic-backend-key")
    for model in models(agent_cls()).values():
        assert model.openai_api_base == "http://studio.invalid:1234/v1"
        assert model.model_name == "hub-model"
        assert model.openai_api_key.get_secret_value() == "synthetic-backend-key"


def test_openai_compatible_backend_uses_its_base_url(settings_env):
    settings_env(LLM_PROVIDER="openai-compatible", OPENAI_BASE_URL="http://llama.invalid:8080/v1",
                 OPENAI_CHAT_MODEL="qwen-local")
    agent = ResearchAgent()
    assert agent.ultra_model.openai_api_base == "http://llama.invalid:8080/v1"
    assert agent.ultra_model.model_name == "qwen-local"


def test_tier_models_can_be_overridden(settings_env):
    settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_CHAT_MODEL="hub-model",
                 AGENT_QUALITY_MODEL="quality-model", AGENT_ULTRA_MODEL="ultra-model")
    devops = DevOpsAgent()
    assert devops.fast_model.model_name == "hub-model"
    assert devops.quality_model.model_name == "quality-model"
    assert ResearchAgent().ultra_model.model_name == "ultra-model"


@pytest.mark.parametrize("effort", [None, "none"])
def test_agents_honour_reasoning_effort(settings_env, effort):
    settings_env(LLM_PROVIDER="lmstudio", **({"LLM_REASONING_EFFORT": effort} if effort else {}))
    for agent in (DevOpsAgent(), ResearchAgent()):
        for model in models(agent).values():
            assert model.reasoning_effort == effort


def test_create_agent_builds_backend_configured_agents(settings_env):
    settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_HOST="http://studio.invalid:1234")
    assert create_agent("research").ultra_model.openai_api_base == "http://studio.invalid:1234/v1"
    with pytest.raises(ValueError):
        create_agent("unknown")
