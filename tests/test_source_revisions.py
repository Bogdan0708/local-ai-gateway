"""File ingestion replaces revisions without affecting independently owned sources."""

import pytest

from src.file_service import FileIngestionService, SecurityError
from src.memory import Document


@pytest.fixture
def ingestion(tmp_path, settings_env, monkeypatch, persistent_memory):
    root = tmp_path / "documents"
    root.mkdir()
    settings_env(DOCUMENTS_PATH=root, CODE_PATH=tmp_path / "code")
    monkeypatch.setattr("src.file_service.get_memory", lambda: persistent_memory)
    return FileIngestionService(chunk_size=100, chunk_overlap=10), persistent_memory, root


def test_edit_replaces_old_file_in_both_searches(ingestion):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    assert service.ingest_file(path) == 1
    path.write_text("# Note\n\nzenith deployment guide")
    assert service.ingest_file(path) == 1
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar") == []
    assert [r.document.content for r in memory.search_bm25("zenith")] == ["# Note\n\nzenith deployment guide"]
    assert all("quasar" not in r.document.content for r in memory.search_semantic("quasar", k=10))


def test_identical_files_keep_separate_source_ownership(ingestion):
    service, memory, root = ingestion
    first, second = root / "first.md", root / "second.md"
    for path in (first, second):
        path.write_text("# Note\n\nquasar launch checklist")
        service.ingest_file(path)
    assert memory.get_stats()["total_documents"] == 2
    first.write_text("# Note\n\nzenith deployment guide")
    service.ingest_file(first)
    assert memory.get_stats()["total_documents"] == 2
    assert [r.document.metadata["source"] for r in memory.search_bm25("quasar")] == [str(second)]
    assert [r.document.metadata["source"] for r in memory.search_bm25("zenith")] == [str(first)]


def test_shorter_revision_removes_old_tail_chunks(ingestion):
    service, memory, root = ingestion
    path = root / "note.txt"
    path.write_text("quasar old instructions and mariner rollback. " * 20)
    assert service.ingest_file(path) > 1
    path.write_text("zenith current instructions")
    assert service.ingest_file(path) == 1
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar") == []


def test_empty_revision_clears_prior_chunks(ingestion):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    path.write_text(" \n\t")
    assert service.ingest_file(path) == 0
    assert not memory.has_source(str(path))
    assert memory.search_bm25("quasar") == []


def test_legacy_record_is_replaced_only_for_its_recorded_source(ingestion):
    service, memory, root = ingestion
    path, other = root / "note.md", root / "other.md"
    path.write_text("# Note\n\nzenith deployment guide")
    memory.add_documents([
        Document("quasar old instructions", {"source": str(path), "file_hash": "old", "chunk_index": 0}, id="legacy_note_0"),
        Document("mariner independent instructions", {"source": str(other), "file_hash": "other", "chunk_index": 0}, id="legacy_other_0"),
    ])
    service.ingest_file(path)
    assert memory._collection.get(ids=["legacy_note_0"])["ids"] == []
    assert memory._collection.get(ids=["legacy_other_0"])["ids"] == ["legacy_other_0"]
    assert memory.search_bm25("quasar") == []
    assert memory.search_bm25("mariner")


def test_failed_embedding_preserves_previous_revision(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    path.write_text("# Note\n\nzenith deployment guide")

    def fail(texts):
        raise RuntimeError("synthetic embedding failure")

    monkeypatch.setattr(memory._embeddings, "embed_batch", fail)
    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        service.ingest_file(path)
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar")
    assert memory.search_bm25("zenith") == []


def test_alias_and_relative_path_share_the_same_file_identity(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    alias = root / "alias.md"
    path.write_text("# Note\n\nquasar launch checklist")
    alias.symlink_to(path)
    service.ingest_file(alias)
    path.write_text("# Note\n\nzenith deployment guide")
    monkeypatch.chdir(root)
    service.ingest_file(__import__('pathlib').Path("note.md"))
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar") == []
    assert [r.document.metadata["source"] for r in memory.search_bm25("zenith")] == [str(path)]


def test_alias_cannot_index_a_blocked_source_filename(ingestion):
    service, memory, root = ingestion
    blocked = root / ".env"
    blocked.write_text("SYNTHETIC FILE POLICY FIXTURE - NO CREDENTIALS")
    alias = root / "allowed.txt"
    alias.symlink_to(blocked)
    with pytest.raises(SecurityError):
        service.ingest_file(alias)
    assert memory.get_stats()["total_documents"] == 0


def test_failed_prune_restores_previous_revision(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    path.write_text("# Note\n\nzenith deployment guide")
    original_delete = memory._collection.delete
    fail = True

    def delete_then_fail(**kwargs):
        nonlocal fail
        result = original_delete(**kwargs)
        if fail:
            fail = False
            raise RuntimeError("synthetic lost delete acknowledgement")
        return result

    monkeypatch.setattr(memory._collection, "delete", delete_then_fail)
    with pytest.raises(RuntimeError, match="synthetic lost delete acknowledgement"):
        service.ingest_file(path)
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar")
    assert memory.search_bm25("zenith") == []


def test_changed_file_cannot_commit_an_obsolete_embedding(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    path.write_text("# Note\n\nzenith deployment guide")
    original_embed = memory._embeddings.embed_batch

    def file_changes_during_embedding(texts):
        vectors = original_embed(texts)
        path.write_text("# Note\n\nmariner newest instructions")
        return vectors

    monkeypatch.setattr(memory._embeddings, "embed_batch", file_changes_during_embedding)
    with pytest.raises(RuntimeError, match="changed"):
        service.ingest_file(path)
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar")
    assert memory.search_bm25("zenith") == []


def test_loader_cannot_commit_intermediate_bytes_after_file_is_reverted(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    current = "# Note\n\nzenith deployment guide"
    path.write_text(current)
    original_loader = service.processor._get_loader

    class ChangingLoader:
        def __init__(self, loader_path):
            self.loader_path = loader_path

        def load(self):
            path.write_text("# Note\n\nmariner intermediate bytes")
            try:
                return original_loader(self.loader_path).load()
            finally:
                path.write_text(current)

    monkeypatch.setattr(service.processor, "_get_loader", lambda loader_path: ChangingLoader(loader_path))
    try:
        service.ingest_file(path)
    except RuntimeError as exc:
        assert "changed" in str(exc)
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("mariner") == []


@pytest.mark.parametrize("wrong_owner", ["other-source", "malformed-owner"])
def test_colliding_id_cannot_overwrite_an_unowned_record(ingestion, wrong_owner):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nzenith deployment guide")
    incoming = list(service.processor.process_file(path))[0]
    metadata = {"source": wrong_owner}
    if wrong_owner == "malformed-owner":
        metadata = {"source": str(path), "file_source_id": "wrong-source-id"}
    memory.add_document(Document("lyra independent instructions", metadata, id=incoming.id))
    with pytest.raises(ValueError, match="own|source"):
        service.ingest_file(path)
    assert memory._collection.get(ids=[incoming.id], include=["documents"])["documents"] == ["lyra independent instructions"]


def test_ambiguous_legacy_relative_source_is_preserved(ingestion, monkeypatch):
    from pathlib import Path

    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nzenith deployment guide")
    memory.add_document(Document("quasar legacy from an unknown cwd", {
        "source": "note.md", "file_hash": "old", "chunk_index": 0,
    }, id="legacy_relative_0"))
    monkeypatch.chdir(root)
    service.ingest_file(Path("note.md"))
    assert memory._collection.get(ids=["legacy_relative_0"])["ids"] == ["legacy_relative_0"]


def test_rollback_restores_exact_metadata_for_reused_ids(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.txt"
    path.write_text("quasar accepted instructions and mariner rollback. " * 12)
    assert service.ingest_file(path) > 1
    before = memory._collection.get(include=["documents", "metadatas"])
    replacement = Document(before["documents"][0], {**before["metadatas"][0], "new_marker": "interrupted"}, id=before["ids"][0])
    original_delete = memory._collection.delete
    fail = True

    def delete_then_fail(**kwargs):
        nonlocal fail
        result = original_delete(**kwargs)
        if fail:
            fail = False
            raise RuntimeError("synthetic prune acknowledgement failure")
        return result

    monkeypatch.setattr(memory._collection, "delete", delete_then_fail)
    with pytest.raises(RuntimeError, match="synthetic prune acknowledgement failure"):
        memory.replace_source(str(path), [replacement])
    after = memory._collection.get(include=["documents", "metadatas"])
    assert {i: (d, m) for i, d, m in zip(after["ids"], after["documents"], after["metadatas"])} == {
        i: (d, m) for i, d, m in zip(before["ids"], before["documents"], before["metadatas"])
    }


def test_empty_capture_cannot_clear_a_restored_nonempty_file(ingestion, monkeypatch):
    from contextlib import contextmanager
    from pathlib import Path

    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar accepted instructions")
    service.ingest_file(path)
    current = "# Note\n\nzenith deployment guide"
    path.write_text(current)
    # Model the measured coarse filesystem timestamps, without relying on sleeps.
    unchanged_stat = path.stat()
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *args, **kwargs: unchanged_stat if self == path else original_stat(self, *args, **kwargs))
    original_capture = service.processor._captured_file

    @contextmanager
    def empty_then_restore(source, filename):
        path.write_text("")
        with original_capture(source, filename) as captured:
            path.write_text(current)
            yield captured

    monkeypatch.setattr(service.processor, "_captured_file", empty_then_restore)
    with pytest.raises(RuntimeError, match="changed"):
        service.ingest_file(path)
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar")


def test_relative_replacement_argument_cannot_claim_a_legacy_source(ingestion):
    _, memory, _ = ingestion
    memory.add_document(Document("quasar unknown legacy cwd", {
        "source": "note.md", "file_hash": "old", "chunk_index": 0,
    }, id="legacy_relative_0"))
    with pytest.raises(ValueError, match="absolute|canonical"):
        memory.replace_source("note.md", [])
    assert memory.search_bm25("quasar")


def test_fallback_loader_metadata_keeps_original_source_not_snapshot(ingestion):
    service, memory, root = ingestion
    service = FileIngestionService(use_smart_chunking=False)
    path = root / "note.txt"
    path.write_text("quasar accepted instructions")
    assert service.ingest_file(path) > 0
    records = memory._collection.get(include=["metadatas"])
    assert records["metadatas"]
    assert all(metadata["source"] == str(path) for metadata in records["metadatas"])


def test_rejected_store_write_does_not_erase_the_accepted_revision(ingestion, monkeypatch):
    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar accepted instructions")
    service.ingest_file(path)
    before = memory._collection.get(include=["documents", "metadatas"])
    path.write_text("# Note\n\nzenith candidate instructions")

    def rejected(**kwargs):
        raise RuntimeError("synthetic write rejection")

    monkeypatch.setattr(memory._collection, "upsert", rejected)
    with pytest.raises(RuntimeError, match="synthetic write rejection"):
        service.ingest_file(path)
    after = memory._collection.get(include=["documents", "metadatas"])
    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]
    assert memory.search_bm25("quasar")


def test_crashed_replacement_recovers_before_existing_client_reads(ingestion):
    import os
    import subprocess
    import sys

    service, memory, root = ingestion
    path = root / "note.md"
    path.write_text("# Note\n\nquasar launch checklist")
    service.ingest_file(path)
    assert memory.search_bm25("quasar")
    path.write_text("# Note\n\nzenith deployment guide")
    script = """
import os, sys
from tests import conftest
from src import file_service
from src.memory import EmbeddingsProvider, HybridMemory
EmbeddingsProvider.embed = lambda self, text: [1.0,0.25,0.5]
memory = HybridMemory(persist_dir=sys.argv[1], enable_reranker=False)
file_service.get_memory = lambda: memory
original_delete = memory._collection.delete
def crash_after_delete(**kwargs):
    original_delete(**kwargs)
    os._exit(23)
memory._collection.delete = crash_after_delete
file_service.FileIngestionService().ingest_file(__import__('pathlib').Path(sys.argv[2]))
"""
    result = subprocess.run([sys.executable, "-c", script, str(memory.persist_dir), str(path)], env=os.environ.copy(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stderr[-2000:]
    assert memory.get_stats()["total_documents"] == 1
    assert memory.search_bm25("quasar")
    assert memory.search_bm25("zenith") == []
