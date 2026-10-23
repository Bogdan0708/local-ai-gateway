"""Exercise backend routing with an in-process transport; no model or GPU calls."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api_gateway import app, call_llm, stream_llm_response
from src.config import Settings


@pytest.fixture
def capture_backend(monkeypatch):
    requests = []
    client_class = httpx.AsyncClient

    def install(status=200):
        def handle(request):
            requests.append(request)
            payload = json.loads(request.content)
            if status != 200:
                return httpx.Response(status, json={"error": "backend unavailable"})
            if payload.get("stream"):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                    'data: [DONE]\n\n',
                )
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            })

        monkeypatch.setattr(
            "src.api_gateway.httpx.AsyncClient",
            lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs),
        )
        return requests

    return install


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("provider,expected_url,expected_model", [
    ("openai-compatible", "http://llama.invalid:8080/local/v1/chat/completions", "qwen-local"),
    ("lmstudio", "http://studio.invalid:1234/v1/chat/completions", "studio-model"),
])
def test_chat_uses_selected_backend(
    settings_env, capture_backend, streaming, provider, expected_url, expected_model,
):
    settings_env(
        LLM_PROVIDER=provider,
        OPENAI_BASE_URL="http://llama.invalid:8080/local/v1/",
        OPENAI_CHAT_MODEL="qwen-local",
        OPENAI_API_KEY="synthetic-backend-key",
        LMSTUDIO_HOST="http://studio.invalid:1234/",
        LMSTUDIO_CHAT_MODEL="studio-model",
    )
    requests = capture_backend()

    async def invoke():
        kwargs = dict(messages=[{"role": "user", "content": "hello"}],
                      model="", temperature=0.2, max_tokens=12)
        if streaming:
            chunks = [chunk async for chunk in stream_llm_response(**kwargs)]
            assert "ok" in chunks[0]
            assert chunks.count("data: [DONE]\n\n") == 1
        else:
            assert await call_llm(**kwargs) == {
                "content": "ok", "prompt_tokens": 3, "completion_tokens": 1,
            }

    asyncio.run(invoke())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer synthetic-backend-key"
    assert json.loads(request.content) == {
        "model": expected_model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "max_tokens": 12,
        **({"stream": True} if streaming else {}),
    }


@pytest.mark.parametrize("streaming", [False, True])
def test_backend_error_is_not_success_or_retried(settings_env, capture_backend, streaming):
    settings_env(LLM_PROVIDER="openai-compatible", OPENAI_BASE_URL="http://llama.invalid/v1")
    requests = capture_backend(status=503)

    async def invoke():
        if streaming:
            return [chunk async for chunk in stream_llm_response([], "qwen-local")]
        return await call_llm([], "qwen-local")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(invoke())
    assert len(requests) == 1


def test_request_model_overrides_backend_default(settings_env, capture_backend):
    settings_env(LLM_PROVIDER="openai-compatible", OPENAI_CHAT_MODEL="default-model")
    requests = capture_backend()
    asyncio.run(call_llm([], "requested-model"))
    assert json.loads(requests[0].content)["model"] == "requested-model"


@pytest.mark.parametrize("url,expected", [
    ("http://server.invalid:8081/v1/", "http://server.invalid:8081"),
    ("http://server.invalid:8081", "http://server.invalid:8081"),
    ("http://server.invalid/proxy/v11", "http://server.invalid/proxy/v11"),
])
def test_host_removes_only_exact_api_suffix(settings_env, url, expected):
    settings = settings_env(LLM_PROVIDER="openai-compatible", OPENAI_BASE_URL=url)
    assert settings.active_host == expected


def test_generic_model_keeps_legacy_fallback():
    settings = Settings(
        _env_file=None,
        jwt_secret="synthetic-jwt-secret-000000000000000000",
        api_key="synthetic-api-secret-000000000000000000",
        llm_provider="openai-compatible",
        openai_chat_model=None,
        lmstudio_chat_model="legacy-alias",
    )
    assert settings.active_chat_model == "legacy-alias"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
def test_omitted_request_model_uses_configured_alias(settings_env, capture_backend, path):
    settings = settings_env(
        LLM_PROVIDER="openai-compatible",
        OPENAI_BASE_URL="http://llama.invalid:8080/v1",
        OPENAI_CHAT_MODEL="qwen-local",
    )
    requests = capture_backend()
    with TestClient(app) as client:
        response = client.post(
            path,
            headers={"X-API-Key": settings.api_key},
            json={"messages": [{"role": "user", "content": "hello"}], "use_rag": False},
        )
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "qwen-local"
    assert len(requests) == 1
    assert str(requests[0].url) == "http://llama.invalid:8080/v1/chat/completions"
    assert json.loads(requests[0].content)["model"] == "qwen-local"


def _bare_settings(**overrides):
    return Settings(
        _env_file=None,
        jwt_secret="synthetic-jwt-secret-000000000000000000",
        api_key="synthetic-api-secret-000000000000000000",
        **overrides,
    )


def test_lmstudio_defaults_match_the_shared_hub():
    settings = _bare_settings()
    assert settings.lmstudio_chat_model == "qwen/qwen3.6-35b-a3b"
    assert settings.lmstudio_embed_model == "text-embedding-nomic-embed-text-v1.5"
    assert settings.llm_reasoning_effort is None


def test_reasoning_effort_rejects_unknown_values():
    with pytest.raises(ValueError):
        _bare_settings(llm_reasoning_effort="maximum")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("effort", [None, "none", "high"])
def test_reasoning_effort_is_sent_only_when_configured(
    settings_env, capture_backend, streaming, effort,
):
    settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_HOST="http://studio.invalid:1234",
                 **({"LLM_REASONING_EFFORT": effort} if effort else {}))
    requests = capture_backend()

    async def invoke():
        if streaming:
            return [chunk async for chunk in stream_llm_response([], "")]
        return await call_llm([], "")

    asyncio.run(invoke())
    payload = json.loads(requests[0].content)
    if effort is None:
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == effort


def test_embeddings_response_reports_the_model_actually_used(settings_env):
    from unittest.mock import MagicMock, patch

    settings = settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_EMBED_MODEL="hub-embedder")
    memory = MagicMock()
    memory._embeddings.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
    with patch("src.api_gateway.get_memory", return_value=memory), TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            headers={"X-API-Key": settings.api_key},
            json={"model": "client-asked-for-this", "input": ["a", "b"]},
        )
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "hub-embedder"
    memory._embeddings.embed_batch.assert_called_once_with(["a", "b"])


def test_chat_created_is_a_real_unix_timestamp_in_any_local_timezone(
    settings_env, capture_backend, monkeypatch,
):
    import time

    monkeypatch.setenv("TZ", "Asia/Tokyo")  # UTC+9: a naive utcnow().timestamp() is 9 h off
    time.tzset()
    try:
        settings = settings_env(LLM_PROVIDER="lmstudio", LMSTUDIO_HOST="http://studio.invalid:1234")
        capture_backend()
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": settings.api_key},
                json={"messages": [{"role": "user", "content": "hi"}], "use_rag": False},
            )
        assert response.status_code == 200, response.text
        assert abs(response.json()["created"] - time.time()) < 60
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
