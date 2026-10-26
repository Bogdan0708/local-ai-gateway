"""Explicit strategies work through the same document ingestion interface."""

import numpy as np
import pytest

from src.chunking import (
    ChunkConfig, ChunkingStrategy, SemanticChunker, chunk_document, chunk_text, get_chunker,
)
from src.file_service import DocumentProcessor


@pytest.fixture
def offline_semantic_model(monkeypatch):
    class Encoder:
        def encode(self, sentences, convert_to_numpy):
            return np.array([[1.0, 0.0] for _ in sentences])

    monkeypatch.setattr(SemanticChunker, "model", property(lambda self: Encoder()))


@pytest.mark.parametrize("strategy", [
    ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC, ChunkingStrategy.STRUCTURE,
])
def test_explicit_strategy_ingests_text_with_source_metadata(
    tmp_path, settings_env, offline_semantic_model, strategy,
):
    settings_env(DOCUMENTS_PATH=tmp_path, CODE_PATH=tmp_path / "code")
    path = tmp_path / "checklist.txt"
    path.write_text("Quasar launch checklist. Mariner rollback instructions.")
    documents = list(DocumentProcessor(chunking_strategy=strategy).process_file(path))
    content = "\n".join(document.content for document in documents)
    assert "Quasar launch checklist" in content
    assert "Mariner rollback instructions" in content
    assert all(document.metadata["source"] == str(path) for document in documents)
    assert all(document.metadata["chunking_strategy"] == strategy.value for document in documents)


@pytest.mark.parametrize("strategy", [
    ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC, ChunkingStrategy.STRUCTURE,
])
def test_factory_chunkers_accept_document_hints(offline_semantic_model, strategy):
    chunker = get_chunker(ChunkConfig(strategy=strategy))
    chunks = chunker.chunk("Quasar launch. Mariner rollback.", content_type="text/plain", filename="note.txt")
    assert chunks
    assert "Quasar launch" in "\n".join(chunk.content for chunk in chunks)


def test_structure_strategy_uses_html_for_html_files(tmp_path, settings_env, monkeypatch):
    settings_env(DOCUMENTS_PATH=tmp_path, CODE_PATH=tmp_path / "code")
    from src.config import get_file_whitelist
    whitelist = get_file_whitelist()
    monkeypatch.setattr(whitelist, "_allowed_extensions", [*whitelist.allowed_extensions, ".html"])
    path = tmp_path / "checklist.html"
    path.write_text("<h1>Checklist</h1><p>quasar launch</p><p>mariner rollback</p>")
    documents = list(DocumentProcessor(chunking_strategy=ChunkingStrategy.STRUCTURE).process_file(path))
    content = "\n".join(document.content for document in documents)
    assert "quasar launch" in content
    assert "mariner rollback" in content
    assert "<p>" not in content


def test_structure_text_helper_keeps_short_markdown():
    chunks = chunk_text("# Checklist\n\nquasar launch", strategy=ChunkingStrategy.STRUCTURE)
    assert "quasar launch" in "\n".join(chunk.content for chunk in chunks)


def test_document_helper_honours_explicit_recursive_strategy():
    chunks = chunk_document("# Checklist\n\nquasar launch", filename="note.md",
                            config=ChunkConfig(strategy=ChunkingStrategy.RECURSIVE))
    assert chunks
    assert all(chunk.metadata["strategy"] == "recursive" for chunk in chunks)


def test_semantic_strategy_can_fall_back_without_model(tmp_path, settings_env, monkeypatch):
    settings_env(DOCUMENTS_PATH=tmp_path, CODE_PATH=tmp_path / "code")
    monkeypatch.setattr(SemanticChunker, "model", property(lambda self: None))
    path = tmp_path / "checklist.txt"
    path.write_text("Quasar launch checklist. Mariner rollback instructions.")
    documents = list(DocumentProcessor(chunking_strategy=ChunkingStrategy.SEMANTIC).process_file(path))
    assert documents
    assert "Quasar launch checklist" in "\n".join(document.content for document in documents)
    assert all(document.metadata["chunking_strategy"] == "recursive" for document in documents)
