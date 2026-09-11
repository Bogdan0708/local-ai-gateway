"""The exported application must actually carry the advertised routes.

A second `app = FastAPI(...)` used to replace the first one after the MCP
router had been mounted, so the shipped app served neither /mcp/* nor
/v1/web/search.
"""

from fastapi.testclient import TestClient

from src.api_gateway import app
from src.config import DEFAULT_ALLOWED_EXTENSIONS, get_file_whitelist
from src.file_service import FileValidator
from tests.conftest import TEST_API_KEY


def _iter_paths(routes):
    """Flatten app.routes, including routers wrapped by FastAPI's lazy include."""
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield path
        nested = getattr(route, "original_router", None) or getattr(route, "router", None)
        if nested is not None:
            yield from _iter_paths(nested.routes)


PATHS = set(_iter_paths(app.routes))


def test_mcp_routes_are_mounted():
    mcp_paths = [path for path in PATHS if path.startswith("/mcp")]
    assert mcp_paths, f"no /mcp routes on the exported app: {sorted(PATHS)}"


def test_mcp_health_route_responds():
    with TestClient(app) as client:
        response = client.get("/mcp/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_mcp_action_routes_require_auth():
    """Restoring the MCP routes must not hand out unauthenticated access."""
    with TestClient(app, client=("203.0.113.7", 1234)) as client:
        response = client.post("/mcp/knowledge/search", json={"query": "anything"})
    assert response.status_code == 401


def test_web_search_route_survives_assembly():
    assert "/v1/web/search" in PATHS


def test_core_routes_are_still_present():
    for path in ("/health", "/api/chat", "/api/read-file", "/v1/chat/completions"):
        assert path in PATHS


def test_single_application_instance():
    """Every registered route belongs to the app we export."""
    assert app.title
    # A duplicate instantiation shows up as duplicated /health registrations.
    health_routes = [path for path in _iter_paths(app.routes) if path == "/health"]
    assert len(health_routes) == 1


def test_file_whitelist_works_on_clean_checkout(tmp_path):
    """No config/file_whitelist.yaml must not mean 'allow nothing'."""
    whitelist = get_file_whitelist()
    assert ".txt" in whitelist.allowed_extensions
    assert set(DEFAULT_ALLOWED_EXTENSIONS) <= set(whitelist.allowed_extensions)

    sample = tmp_path / "sample.txt"
    sample.write_text("synthetic")
    FileValidator().validate_extension(sample)  # must not raise


def test_blocked_patterns_cover_secrets():
    patterns = set(get_file_whitelist().blocked_patterns)
    assert ".env" in patterns
    assert "*.pem" in patterns
    assert any(p.startswith("id_rsa") for p in patterns)


def test_mcp_ingest_and_search_round_trip(settings_env):
    """The restored MCP routes must actually work, not just be registered."""
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    from unittest.mock import MagicMock, patch

    from src.memory import Document, SearchResult

    memory = MagicMock()
    memory.add_documents.return_value = ["doc-0"]
    memory.search.return_value = [
        SearchResult(
            document=Document(content="synthetic chunk", metadata={"source": "test"}),
            score=0.9,
            source="hybrid",
        )
    ]

    headers = {"X-API-Key": TEST_API_KEY}
    with patch("src.memory.get_memory", return_value=memory):
        with TestClient(app, client=("203.0.113.7", 1234)) as client:
            ingest = client.post(
                "/mcp/knowledge/ingest",
                json={
                    "content": "Synthetic ingestion content. " * 40,
                    "metadata": {"source": "test", "filename": "note.md"},
                    "chunk_strategy": "fixed",
                },
                headers=headers,
            )
            search = client.post(
                "/mcp/knowledge/search",
                json={"query": "synthetic", "top_k": 3, "use_reranking": False},
                headers=headers,
            )

    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["chunks_created"] >= 1
    assert ingest.json()["document_id"] == "doc-0"
    memory.add_documents.assert_called_once()

    assert search.status_code == 200, search.text
    assert search.json()["total"] == 1
    assert search.json()["results"][0]["content"] == "synthetic chunk"


def test_mcp_agent_invoke_passes_the_right_arguments(settings_env):
    """The research agent takes (query, sources); devops takes (task)."""
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    from unittest.mock import AsyncMock, MagicMock, patch

    headers = {"X-API-Key": TEST_API_KEY}
    devops, research = MagicMock(), MagicMock()
    devops.run = AsyncMock(return_value={"status": "completed", "code": "x"})
    research.run = AsyncMock(return_value={"status": "completed", "synthesis": "y"})

    def fake_create_agent(agent_type, *args, **kwargs):
        return devops if agent_type == "devops" else research

    with patch("src.agents.create_agent", side_effect=fake_create_agent):
        with TestClient(app, client=("203.0.113.7", 1234)) as client:
            devops_response = client.post(
                "/mcp/agents/invoke",
                json={"agent_type": "devops", "task": "write a parser"},
                headers=headers,
            )
            research_response = client.post(
                "/mcp/agents/invoke",
                json={
                    "agent_type": "research",
                    "task": "summarise the notes",
                    "sources": ["doc-a", "doc-b"],
                },
                headers=headers,
            )

    assert devops_response.status_code == 200, devops_response.text
    devops.run.assert_awaited_once_with("write a parser")

    assert research_response.status_code == 200, research_response.text
    research.run.assert_awaited_once_with("summarise the notes", ["doc-a", "doc-b"])


def test_mcp_research_agent_without_sources_still_works(settings_env):
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    from unittest.mock import AsyncMock, MagicMock, patch

    research = MagicMock()
    research.run = AsyncMock(return_value={"status": "completed", "synthesis": "y"})

    with patch("src.agents.create_agent", return_value=research):
        with TestClient(app, client=("203.0.113.7", 1234)) as client:
            response = client.post(
                "/mcp/agents/invoke",
                json={"agent_type": "research", "task": "summarise"},
                headers={"X-API-Key": TEST_API_KEY},
            )

    assert response.status_code == 200, response.text
    research.run.assert_awaited_once_with("summarise", [])
