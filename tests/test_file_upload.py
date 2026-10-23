"""POST /v1/files: uploads pass the documents-dir path policy and leave no staged files."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api_gateway import app


@pytest.fixture
def upload(settings_env, tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    settings = settings_env(DOCUMENTS_PATH=str(documents), CODE_PATH=str(tmp_path / "code"))
    memory = MagicMock()

    # Longer than the Markdown chunker's 100-character minimum, so it yields chunks.
    note = (b"# Shared hub\n\n" + b"LM Studio serves the shared inference hub on port 1234 "
            b"for every agent running in WSL on this laptop. " * 4)

    def send(filename, content=note):
        with patch("src.file_service.get_memory", return_value=memory), TestClient(app) as client:
            return client.post(
                "/v1/files",
                headers={"X-API-Key": settings.api_key},
                files={"file": (filename, content, "text/markdown")},
            )

    return send, memory, documents


def staged_files(documents):
    return [p for p in documents.rglob("*") if p.is_file()]


def test_upload_is_indexed_under_its_original_name(upload):
    send, memory, documents = upload
    response = send("hub-note.md")
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] >= 1
    sources = [d.metadata.get("source", "") for d in memory.add_documents.call_args.args[0]]
    assert sources and all(s.endswith("hub-note.md") for s in sources)
    assert staged_files(documents) == []


def test_upload_filename_cannot_escape_the_staging_dir(upload, tmp_path):
    send, memory, documents = upload
    response = send("../../escape.md")
    assert response.status_code == 200, response.text
    assert not (tmp_path / "escape.md").exists()
    sources = [d.metadata.get("source", "") for d in memory.add_documents.call_args.args[0]]
    assert all(str(documents) in s for s in sources)
    assert staged_files(documents) == []


@pytest.mark.parametrize("filename", ["payload.exe", ".env", ""])
def test_rejected_upload_is_not_indexed_and_leaves_nothing_behind(upload, filename):
    send, memory, documents = upload
    response = send(filename)
    assert response.status_code in (400, 422), response.text
    memory.add_documents.assert_not_called()
    assert staged_files(documents) == []
