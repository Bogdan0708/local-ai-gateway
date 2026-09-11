"""Path and filename boundary tests for /api/read-file.

The endpoint must compare real filesystem ancestry (resolve + is_relative_to),
not string prefixes, and must apply the same filename policy the ingestion
path uses.
"""

import pytest
from fastapi.testclient import TestClient

from src.api_gateway import app
from tests.conftest import TEST_API_KEY

REMOTE = ("203.0.113.7", 1234)
AUTH = {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def fixtures(tmp_path, settings_env, monkeypatch):
    """Build an allowed root plus the traps around it."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "sample.txt").write_text("synthetic allowed content")
    (allowed / ".env").write_text("SYNTHETIC ENV FIXTURE - NO CREDENTIALS")
    (allowed / "key.pem").write_text("SYNTHETIC PEM FIXTURE - NOT A KEY")

    (tmp_path / "outside.txt").write_text("SYNTHETIC OUTSIDE-ROOT FIXTURE")

    sibling = tmp_path / "allowed-evil"
    sibling.mkdir()
    (sibling / "x.txt").write_text("SYNTHETIC SIBLING-PREFIX FIXTURE")

    escape = allowed / "escape.txt"
    escape.symlink_to(tmp_path / "outside.txt")

    monkeypatch.setenv("ALLOWED_PATH_PREFIXES", str(allowed))
    settings_env(ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    return tmp_path


def _read(path):
    with TestClient(app, client=REMOTE) as client:
        return client.post("/api/read-file", json={"path": str(path)}, headers=AUTH)


def test_allowed_file_is_readable(fixtures):
    response = _read(fixtures / "allowed" / "sample.txt")
    assert response.status_code == 200
    assert response.json()["content"] == "synthetic allowed content"


def test_traversal_out_of_root_is_rejected(fixtures):
    response = _read(f"{fixtures / 'allowed'}/../outside.txt")
    assert response.status_code == 403


def test_sibling_prefix_directory_is_rejected(fixtures):
    response = _read(fixtures / "allowed-evil" / "x.txt")
    assert response.status_code == 403


def test_symlink_escaping_root_is_rejected(fixtures):
    response = _read(fixtures / "allowed" / "escape.txt")
    assert response.status_code == 403


def test_dotenv_inside_root_is_rejected(fixtures):
    response = _read(fixtures / "allowed" / ".env")
    assert response.status_code == 403


def test_pem_inside_root_is_rejected(fixtures):
    response = _read(fixtures / "allowed" / "key.pem")
    assert response.status_code == 403


def test_missing_file_is_not_found(fixtures):
    response = _read(fixtures / "allowed" / "nope.txt")
    assert response.status_code == 404
