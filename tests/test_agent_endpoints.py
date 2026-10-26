"""Agent endpoints accept the flat request bodies their docstrings document."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api_gateway import app
from src.auth import require_auth

client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_auth():
    """Override auth for this module only; never leak into other test files."""
    app.dependency_overrides[require_auth] = lambda: {"sub": "test", "scopes": ["*"]}
    yield
    app.dependency_overrides.pop(require_auth, None)


def test_research_agent_accepts_documented_flat_body_with_sources():
    run = AsyncMock(return_value={"query": "q", "synthesis": "s", "sources_used": 2,
                                  "status": "completed"})
    with patch("src.agents.ResearchAgent") as agent_cls:
        agent_cls.return_value.run = run
        response = client.post("/v1/agents/research", json={
            "task": "What changed?",
            "sources": ["source one", "source two"],
        })

    assert response.status_code == 200, response.text
    run.assert_awaited_once_with(query="What changed?", sources=["source one", "source two"])
    assert response.json()["task"] == "What changed?"


def test_research_agent_sources_default_to_empty():
    run = AsyncMock(return_value={"status": "completed"})
    with patch("src.agents.ResearchAgent") as agent_cls:
        agent_cls.return_value.run = run
        response = client.post("/v1/agents/research", json={"task": "What changed?"})

    assert response.status_code == 200, response.text
    run.assert_awaited_once_with(query="What changed?", sources=[])


def test_devops_agent_accepts_documented_flat_body():
    run = AsyncMock(return_value={"status": "completed", "code": "pass"})
    with patch("src.agents.DevOpsAgent") as agent_cls:
        agent_cls.return_value.run = run
        response = client.post("/v1/agents/devops", json={
            "task": "Write a function",
            "agent_type": "devops",
        })

    assert response.status_code == 200, response.text
    run.assert_awaited_once_with(task="Write a function")
