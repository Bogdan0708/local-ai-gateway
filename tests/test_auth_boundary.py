"""Authentication boundary tests for require_local_or_auth.

The implicit `local_user` identity must only be granted when the request
really arrives on the loopback interface AND the operator opted in via
ALLOW_LOOPBACK_UNAUTHENTICATED. Everything else must authenticate.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api_gateway import app
from tests.conftest import TEST_API_KEY

REMOTE = ("203.0.113.7", 1234)
LOOPBACK = ("127.0.0.1", 50000)


@pytest.fixture
def readable_file(tmp_path, settings_env, monkeypatch):
    """An allowed, readable file so a successful call returns 200."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "sample.txt"
    target.write_text("synthetic fixture content")
    monkeypatch.setenv("ALLOWED_PATH_PREFIXES", str(allowed))
    return target


def _read_file(client_addr, path, headers=None):
    with TestClient(app, client=client_addr) as client:
        return client.post(
            "/api/read-file", json={"path": str(path)}, headers=headers or {}
        )


def test_remote_unauthenticated_is_rejected(readable_file, settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="true", HOST="127.0.0.1")
    response = _read_file(REMOTE, readable_file)
    assert response.status_code == 401


def test_loopback_without_flag_is_rejected(readable_file, settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    response = _read_file(LOOPBACK, readable_file)
    assert response.status_code == 401


def test_loopback_with_flag_is_allowed(readable_file, settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="true", HOST="127.0.0.1")
    response = _read_file(LOOPBACK, readable_file)
    assert response.status_code == 200
    assert response.json()["content"] == "synthetic fixture content"


def test_flag_is_ignored_when_not_bound_to_loopback(readable_file, settings_env):
    """A wildcard bind makes proxied remote traffic look local, so the flag
    must not be honoured."""
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="true", HOST="0.0.0.0")
    assert _read_file(LOOPBACK, readable_file).status_code == 401
    assert (
        _read_file(LOOPBACK, readable_file, {"X-API-Key": TEST_API_KEY}).status_code
        == 200
    )


def test_remote_with_api_key_is_allowed(readable_file, settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    response = _read_file(REMOTE, readable_file, {"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200


def test_remote_with_invalid_api_key_is_rejected(readable_file, settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    response = _read_file(REMOTE, readable_file, {"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_loopback_default_is_closed(readable_file, settings_env, monkeypatch):
    """Unset flag must behave as false."""
    monkeypatch.delenv("ALLOW_LOOPBACK_UNAUTHENTICATED", raising=False)
    settings_env()
    response = _read_file(LOOPBACK, readable_file)
    assert response.status_code == 401


def test_chat_endpoint_requires_auth(settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    app.dependency_overrides.clear()
    with patch("src.api_gateway.call_llm") as mock_llm:
        mock_llm.return_value = {
            "content": "unused",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        with TestClient(app, client=REMOTE) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "use_rag": False,
                },
            )
    assert response.status_code == 401
    mock_llm.assert_not_called()
