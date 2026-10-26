"""Real persistent vector storage and keyword retrieval must expose the same IDs."""

import pytest

from src.memory import Document, EmbeddingsProvider, HybridMemory


@pytest.fixture
def memory(tmp_path, monkeypatch):
    # Replace only external inference; both Chroma storage and BM25 are real.
    monkeypatch.setattr(EmbeddingsProvider, "embed", lambda self, text: [1.0, 0.25, 0.5])
    instance = HybridMemory(persist_dir=tmp_path / "chroma", enable_reranker=False)
    instance.add_documents([
        Document("blue ocean tide", {"source": "ocean"}, id="ocean"),
        Document("green forest trees", {"source": "forest"}, id="forest"),
        Document("violin orchestra music", {"source": "music"}, id="music"),
    ])
    try:
        yield instance
    finally:
        instance.close()


def test_reingestion_does_not_duplicate_keyword_results(memory):
    note = Document("quasar launch checklist", {"source": "note"}, id="note")
    memory.add_document(note)
    memory.add_document(note)
    assert memory.get_stats()["total_documents"] == 4
    assert [result.document.id for result in memory.search_bm25("quasar")] == ["note"]


def test_updated_id_has_same_content_in_both_indexes(memory):
    memory.add_document(Document("quasar launch checklist", {"source": "old"}, id="note"))
    memory.add_document(Document("zenith deployment guide", {"source": "new"}, id="note"))
    persisted = memory._collection.get(ids=["note"], include=["documents", "metadatas"])
    assert persisted["documents"] == ["zenith deployment guide"]
    keyword = memory.search_bm25("zenith")
    assert [(r.document.id, r.document.content, r.document.metadata["source"]) for r in keyword] == [
        ("note", "zenith deployment guide", "new"),
    ]
    assert memory.search_bm25("quasar") == []
    semantic = [r.document for r in memory.search_semantic("zenith", k=4) if r.document.id == "note"]
    assert [(d.content, d.metadata["source"]) for d in semantic] == [("zenith deployment guide", "new")]


def test_deletion_removes_keyword_results_without_restart(memory):
    memory.add_document(Document("quasar launch checklist", {"source": "note"}, id="note"))
    assert memory.search_bm25("quasar")
    assert memory.delete_document("note")
    assert memory.search_bm25("quasar") == []
    assert "note" not in [r.document.id for r in memory.search_semantic("quasar", k=4)]


def test_duplicate_ids_in_one_batch_use_the_latest_content(memory):
    documents = [
        Document("quasar old guide", {"source": "old"}, id="note"),
        Document("zenith current guide", {"source": "current"}, id="note"),
    ]
    ids = memory.add_documents(documents)
    assert ids == ["note"]
    assert memory._collection.get(ids=["note"], include=["documents"])["documents"] == ["zenith current guide"]
    assert [r.document.content for r in memory.search_bm25("zenith")] == ["zenith current guide"]
    assert memory.search_bm25("quasar") == []


def test_failed_keyword_refresh_does_not_serve_deleted_content(memory, monkeypatch):
    memory.add_document(Document("quasar launch checklist", {"source": "note"}, id="note"))
    assert memory.search_bm25("quasar")
    assert memory.delete_document("note")
    original = memory._collection.get
    fail = True

    def transient_failure(*args, **kwargs):
        nonlocal fail
        if fail:
            fail = False
            raise RuntimeError("synthetic snapshot failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(memory._collection, "get", transient_failure)
    with pytest.raises(RuntimeError, match="synthetic snapshot failure"):
        memory.search_bm25("quasar")
    assert memory.search_bm25("quasar") == []


def test_punctuation_only_corpus_does_not_break_keyword_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(EmbeddingsProvider, "embed", lambda self, text: [1.0, 0.25, 0.5])
    instance = HybridMemory(persist_dir=tmp_path / "punctuation", enable_reranker=False)
    try:
        instance.add_document(Document("--- !!!", {"source": "note"}, id="note"))
        assert instance.search_bm25("anything") == []
        assert instance.get_stats()["total_documents"] == 1
    finally:
        instance.close()


def test_fresh_knowledge_base_returns_matching_keyword_document(persistent_memory):
    persistent_memory.add_document(Document("quasar launch checklist", {"source": "note"}, id="note"))
    assert [r.document.id for r in persistent_memory.search_bm25("quasar")] == ["note"]
    assert persistent_memory.search_bm25("absent") == []


def test_equal_document_frequency_does_not_hide_keyword_match(persistent_memory):
    persistent_memory.add_documents([
        Document("quasar launch checklist", {"source": "note"}, id="note"),
        Document("mariner rollback guide", {"source": "other"}, id="other"),
    ])
    assert [r.document.id for r in persistent_memory.search_bm25("quasar")] == ["note"]
