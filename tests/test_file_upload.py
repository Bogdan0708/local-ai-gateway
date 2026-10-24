"""POST /v1/files retains indexed sources and cleans up rejected uploads."""

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
    sources = [d.metadata.get("source", "") for d in memory.replace_source.call_args.args[1]]
    assert sources and all(s.endswith("hub-note.md") for s in sources)
    assert len(staged_files(documents)) == 1


def test_upload_filename_cannot_escape_the_staging_dir(upload, tmp_path):
    send, memory, documents = upload
    response = send("../../escape.md")
    assert response.status_code == 200, response.text
    assert not (tmp_path / "escape.md").exists()
    sources = [d.metadata.get("source", "") for d in memory.replace_source.call_args.args[1]]
    assert all(str(documents) in s for s in sources)
    assert len(staged_files(documents)) == 1


@pytest.mark.parametrize("filename", ["payload.exe", ".env", ""])
def test_rejected_upload_is_not_indexed_and_leaves_nothing_behind(upload, filename):
    send, memory, documents = upload
    response = send(filename)
    assert response.status_code in (400, 422), response.text
    memory.replace_source.assert_not_called()
    assert staged_files(documents) == []


def test_successful_upload_keeps_its_source_readable(upload):
    from pathlib import Path

    send, memory, documents = upload
    response = send("durable-note.md")
    assert response.status_code == 200, response.text
    indexed = memory.replace_source.call_args.args[1]
    assert indexed
    source = Path(indexed[0].metadata["source"])
    assert source.is_relative_to(documents)
    assert source.is_file()
    assert source.read_text().startswith("# Shared hub")


def test_short_markdown_upload_is_actually_indexed(upload):
    send, memory, _ = upload
    response = send("short.md", b"# Note\n\nKeep the quasar launch checklist.")
    assert response.status_code == 200, response.text
    assert response.json()["chunks"] > 0
    indexed = memory.replace_source.call_args.args[1]
    assert any("quasar launch checklist" in doc.content for doc in indexed)


def test_empty_upload_is_rejected_and_removed(upload):
    send, memory, documents = upload
    response = send("empty.md", b" \n\t")
    assert response.status_code == 400, response.text
    assert all(not call.args[1] for call in memory.replace_source.call_args_list)
    assert staged_files(documents) == []


@pytest.fixture
def stored_upload(persistent_memory, settings_env, tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    settings = settings_env(DOCUMENTS_PATH=documents, CODE_PATH=tmp_path / "code", ALLOW_LOOPBACK_UNAUTHENTICATED="false")
    monkeypatch.setenv("ALLOWED_PATH_PREFIXES", str(documents))
    monkeypatch.setattr("src.file_service.get_memory", lambda: persistent_memory)
    monkeypatch.setattr("src.api_gateway.get_memory", lambda: persistent_memory)
    headers = {"X-API-Key": settings.api_key}
    with TestClient(app, client=("203.0.113.7", 1234)) as client:
        yield client, headers, documents, persistent_memory


def test_uploaded_source_can_be_listed_and_read(stored_upload):
    from pathlib import Path

    client, headers, documents, memory = stored_upload
    content = b"# Checklist\n\nKeep the quasar launch checklist."
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", content, "text/markdown")})
    assert response.status_code == 200, response.text
    listing = client.get("/v1/files", headers=headers)
    assert listing.status_code == 200
    source = Path(listing.json()[0]["path"])
    assert source.is_relative_to(documents)
    assert source.read_bytes() == content
    read = client.post("/api/read-file", headers=headers, json={"path": str(source)})
    assert read.status_code == 200, read.text
    assert read.json()["content"] == content.decode()
    assert memory.get_stats()["total_documents"] > 0


def test_upload_error_with_pending_recovery_preserves_indexed_source(stored_upload, monkeypatch):
    from pathlib import Path

    client, headers, _, memory = stored_upload
    original = memory._collection.upsert
    original_delete = memory._collection.delete

    def commit_then_fail(**kwargs):
        original(**kwargs)
        raise RuntimeError("synthetic lost acknowledgement after commit")

    def unavailable_rollback(**kwargs):
        raise RuntimeError("synthetic rollback outage")

    monkeypatch.setattr(memory._collection, "upsert", commit_then_fail)
    monkeypatch.setattr(memory._collection, "delete", unavailable_rollback)
    content = b"# Checklist\n\nKeep the quasar launch checklist."
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", content, "text/markdown")})
    assert response.status_code == 400, response.text
    results = memory._collection.get(include=["metadatas"])
    assert len(results["ids"]) == 1
    assert Path(results["metadatas"][0]["source"]).read_bytes() == content
    monkeypatch.setattr(memory._collection, "delete", original_delete)
    assert memory.search_bm25("deleted") == []
    assert memory.get_stats()["total_documents"] == 0


def test_upload_error_before_storage_commit_removes_new_file(stored_upload, monkeypatch):
    client, headers, documents, memory = stored_upload

    def reject(**kwargs):
        raise RuntimeError("synthetic storage rejection")

    monkeypatch.setattr(memory._collection, "upsert", reject)
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", b"# Checklist\n\nquasar launch", "text/markdown")})
    assert response.status_code == 400
    assert memory.get_stats()["total_documents"] == 0
    assert staged_files(documents) == []


def test_unavailable_commit_check_keeps_source_for_recovery(stored_upload, monkeypatch):
    client, headers, documents, memory = stored_upload
    original_upsert = memory._collection.upsert
    original_get = memory._collection.get

    def commit_then_fail(**kwargs):
        original_upsert(**kwargs)
        raise RuntimeError("synthetic lost acknowledgement")

    def fail_source_check(*args, **kwargs):
        if "where" in kwargs:
            raise RuntimeError("synthetic lookup outage")
        return original_get(*args, **kwargs)

    monkeypatch.setattr(memory._collection, "upsert", commit_then_fail)
    monkeypatch.setattr(memory._collection, "get", fail_source_check)
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", b"# Checklist\n\nquasar launch", "text/markdown")})
    assert response.status_code == 400
    assert len(staged_files(documents)) == 1
    assert memory.get_stats()["total_documents"] == 0


def test_failed_new_upload_preserves_existing_source(stored_upload, monkeypatch):
    from pathlib import Path

    client, headers, documents, memory = stored_upload
    content = b"# Checklist\n\nquasar launch"
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", content, "text/markdown")})
    assert response.status_code == 200
    source = Path(client.get("/v1/files", headers=headers).json()[0]["path"])

    def reject(**kwargs):
        raise RuntimeError("synthetic storage rejection")

    monkeypatch.setattr(memory._collection, "upsert", reject)
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", b"# Changed\n\nmariner rollback", "text/markdown")})
    assert response.status_code == 400
    assert source.read_bytes() == content
    assert staged_files(documents) == [source]


@pytest.mark.parametrize("linked_uploads", [False, True])
def test_file_watcher_cannot_index_a_partial_upload(stored_upload, monkeypatch, linked_uploads):
    import os
    from pathlib import Path
    from src.file_service import DocumentProcessor, FileWatcherHandler

    client, headers, documents, memory = stored_upload
    if linked_uploads:
        relocated = documents / "relocated"
        relocated.mkdir(parents=True)
        (documents / "uploads").symlink_to(relocated, target_is_directory=True)
    handler = FileWatcherHandler(DocumentProcessor(), memory)
    original_fdopen = os.fdopen

    class ObservedWriter:
        def __init__(self, fd, mode):
            self.path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            self.handle = original_fdopen(fd, mode)

        def __enter__(self):
            return self

        def write(self, data):
            middle = len(data) // 2
            self.handle.write(data[:middle])
            self.handle.flush()
            # Watchdog can deliver an event before the upload finishes writing.
            handler._process_file(self.path)
            return middle + self.handle.write(data[middle:])

        def __exit__(self, *args):
            self.handle.close()

    def observe_upload_only(fd, mode, *args, **kwargs):
        if mode == "wb":
            return ObservedWriter(fd, mode)
        return original_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", observe_upload_only)
    content = b"# Checklist\n\n" + b"quasar launch checklist and mariner rollback. " * 6
    response = client.post("/v1/files", headers=headers, files={"file": ("checklist.md", content, "text/markdown")})
    assert response.status_code == 200, response.text
    records = memory._collection.get(include=["documents"])
    assert len(records["ids"]) == response.json()["chunks"]
    assert "mariner rollback" in "\n".join(records["documents"])
    assert len(staged_files(documents)) == 1


def test_watcher_still_indexes_ordinary_files_under_uploads(stored_upload):
    from src.file_service import DocumentProcessor, FileWatcherHandler

    _, _, documents, memory = stored_upload
    ordinary = documents / "uploads" / "manual" / "note.md"
    ordinary.parent.mkdir(parents=True)
    ordinary.write_text("# Checklist\n\nquasar launch checklist")
    FileWatcherHandler(DocumentProcessor(), memory)._process_file(ordinary)
    assert memory.get_stats()["total_documents"] == 1
    assert [r.document.metadata["source"] for r in memory.search_bm25("quasar")] == [str(ordinary)]


@pytest.mark.parametrize("endpoint", ["/health", "/v1/files"])
def test_public_metadata_reads_recover_before_exposing_a_revision(stored_upload, monkeypatch, endpoint):
    from src.file_service import FileIngestionService

    client, headers, documents, memory = stored_upload
    documents.mkdir(parents=True)
    path = documents / "note.md"
    path.write_text("# Note\n\nquasar accepted instructions")
    service = FileIngestionService()
    service.ingest_file(path)
    path.write_text("# Note\n\nzenith uncommitted instructions")
    original_delete = memory._collection.delete

    def fail_cleanup(**kwargs):
        raise RuntimeError("synthetic cleanup outage")

    monkeypatch.setattr(memory._collection, "delete", fail_cleanup)
    with pytest.raises(RuntimeError, match="synthetic cleanup outage"):
        service.ingest_file(path)
    assert memory._collection.count() == 2
    monkeypatch.setattr(memory._collection, "delete", original_delete)
    response = client.get(endpoint, headers=headers)
    assert response.status_code == 200, response.text
    if endpoint == "/health":
        assert response.json()["memory_documents"] == 1
    else:
        assert len(response.json()) == 1
        assert response.json()[0]["chunks"] == 1
