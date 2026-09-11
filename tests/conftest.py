"""Shared pytest fixtures.

Provides safe, synthetic defaults for the settings the application requires so
the suite runs from a clean checkout without a .env file.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Synthetic, non-production values. Set before `src` is imported anywhere.
_TEST_ROOT = Path(tempfile.gettempdir()) / "local-ai-gateway-test"
_DEFAULT_ENV = {
    "JWT_SECRET": "test-jwt-secret-not-a-real-secret-0000000000000000",
    "API_KEY": "test-api-key-not-a-real-secret-00000000000000000000",
    "CHROMA_PERSIST_DIR": str(_TEST_ROOT / "chroma"),
    "DOCUMENTS_PATH": str(_TEST_ROOT / "documents"),
    "CODE_PATH": str(_TEST_ROOT / "code"),
    "ANONYMIZED_TELEMETRY": "False",
}

for _key, _value in _DEFAULT_ENV.items():
    os.environ.setdefault(_key, _value)

TEST_API_KEY = os.environ["API_KEY"]


@pytest.fixture
def settings_env(monkeypatch):
    """Set environment variables and rebuild the cached Settings instance."""
    from src.config import get_settings

    def _apply(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        get_settings.cache_clear()
        return get_settings()

    yield _apply

    get_settings.cache_clear()
