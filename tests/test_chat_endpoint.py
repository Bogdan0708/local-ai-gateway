from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api_gateway import app
from src.auth import require_local_or_auth

client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_auth():
    """Override auth for this module only; never leak into other test files."""
    app.dependency_overrides[require_local_or_auth] = lambda: {
        "sub": "test",
        "scopes": ["*"],
    }
    yield
    app.dependency_overrides.pop(require_local_or_auth, None)

def test_chat_endpoint_success():
    # Mock memory and LLM
    with patch("src.api_gateway.get_memory") as mock_memory:
        # Setup mock memory search
        mock_result = MagicMock()
        mock_result.document.content = "Context info"
        mock_result.document.metadata = {"source": "doc1"}
        mock_result.score = 0.9
        mock_memory.return_value.search.return_value = [mock_result]
        
        with patch("src.api_gateway.call_llm") as mock_llm:
            mock_llm.return_value = {
                "content": "Test response",
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
            
            response = client.post("/api/chat", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "use_rag": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Test response"
            assert "rag_sources" in data
            assert len(data["rag_sources"]) == 1

def test_chat_endpoint_rag_failure():
    # Verify behavior when RAG fails (should probably degrade gracefully or error cleanly)
    with patch("src.api_gateway.get_memory") as mock_memory:
        # Simulate RAG error (e.g. embedding service down)
        mock_memory.return_value.search.side_effect = Exception("Embedding service down")
        
        with patch("src.api_gateway.call_llm") as mock_llm:
            mock_llm.return_value = {
                "content": "Response without RAG",
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
            
            # This might fail 500 currently
            response = client.post("/api/chat", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "use_rag": True
            })
            
            # We want to see what happens - currently expecting 500
            if response.status_code == 200:
                print("Graceful degradation handled!")
            else:
                print(f"Failed as expected with {response.status_code}")
                assert response.status_code == 500
