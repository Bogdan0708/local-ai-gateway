"""Existing memory clients observe externally persisted keyword changes."""

import os
import subprocess
import sys

import pytest

from src.memory import Document, HybridMemory


@pytest.mark.parametrize("operation", ["insert", "update", "delete"])
def test_other_memory_client_refreshes_keyword_results(persistent_memory, operation):
    gateway = persistent_memory
    gateway.add_document(Document("quasar old instructions", {"source": "note"}, id="note"))
    assert gateway.search_bm25("quasar")
    other = HybridMemory(persist_dir=gateway.persist_dir, enable_reranker=False)
    try:
        if operation == "insert":
            other.add_document(Document("zenith new instructions", {"source": "new"}, id="new"))
        elif operation == "update":
            other.add_document(Document("zenith current instructions", {"source": "note"}, id="note"))
        else:
            assert other.delete_document("note")
        if operation == "delete":
            assert gateway.search_bm25("quasar") == []
        else:
            assert gateway.search_bm25("zenith")
            if operation == "update":
                assert gateway.search_bm25("quasar") == []
    finally:
        other.close()


def test_separate_process_write_is_visible_without_gateway_restart(persistent_memory):
    gateway = persistent_memory
    gateway.add_document(Document("quasar old instructions", {"source": "note"}, id="note"))
    assert gateway.search_bm25("quasar")
    script = """
import sys
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('local_ai_knowledge')
collection.upsert(ids=['note'], documents=['zenith current instructions'], metadatas=[{'source':'note'}], embeddings=[[1.0,0.25,0.5]])
"""
    result = subprocess.run([sys.executable, "-c", script, str(gateway.persist_dir)], env=os.environ.copy(), capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr[-2000:]
    assert gateway._collection.get(ids=["note"], include=["documents"])["documents"] == ["zenith current instructions"]
    assert [r.document.content for r in gateway.search_bm25("zenith")] == ["zenith current instructions"]
    assert gateway.search_bm25("quasar") == []
