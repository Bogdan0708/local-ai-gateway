"""
Memory layer with ChromaDB for vector storage and hybrid search.

Implements:
- Semantic search using Ollama embeddings
- BM25 keyword search for hybrid retrieval
- Reciprocal Rank Fusion (RRF) for combining results
- Cross-encoder reranking for improved accuracy (+20-35%)
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from .config import get_settings
from .reranker import CrossEncoderReranker, RerankerConfig, get_reranker

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict[str, Any]
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            # Generate deterministic ID from content hash
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """A search result with relevance score."""

    document: Document
    score: float
    source: str  # "semantic", "bm25", or "hybrid"


class EmbeddingsProvider:
    """
    Generate embeddings using Ollama or LM Studio (OpenAI-compatible) API.

    Supports:
    - Ollama: /api/embeddings endpoint
    - LM Studio: /v1/embeddings endpoint (OpenAI-compatible)
    """

    def __init__(self, host: str, model: str, provider: str = "ollama"):
        self.host = host.rstrip("/")
        self.model = model
        self.provider = provider
        self._client = httpx.Client(timeout=60.0)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if self.provider == "ollama":
            return self._embed_ollama(text)
        else:
            # LM Studio / OpenAI-compatible
            return self._embed_openai_compatible(text)

    def _embed_ollama(self, text: str) -> list[float]:
        """Ollama-specific embedding endpoint."""
        response = self._client.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def _embed_openai_compatible(self, text: str) -> list[float]:
        """OpenAI-compatible embedding endpoint (LM Studio, etc.)."""
        response = self._client.post(
            f"{self.host}/v1/embeddings",
            json={"model": self.model, "input": text},
            headers={"Authorization": "Bearer lm-studio"},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # For OpenAI-compatible, we could batch, but for simplicity use sequential
        return [self.embed(text) for text in texts]

    def close(self):
        self._client.close()


# Alias for backward compatibility
OllamaEmbeddings = EmbeddingsProvider


class BM25Index:
    """BM25 index for keyword-based search."""

    def __init__(self):
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index: Optional[BM25Okapi] = None

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the BM25 index."""
        for doc in documents:
            self._documents.append(doc)
            tokens = self._tokenize(doc.content)
            self._tokenized_corpus.append(tokens)

        # Rebuild index
        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search using BM25."""
        if not self._index or not self._documents:
            return []

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        # Get top-k results
        scored_docs = list(zip(self._documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:k]:
            if score > 0:
                results.append(SearchResult(document=doc, score=score, source="bm25"))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re

        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def clear(self) -> None:
        """Clear the index."""
        self._documents = []
        self._tokenized_corpus = []
        self._index = None


class HybridMemory:
    """
    Hybrid memory combining ChromaDB (semantic) and BM25 (keyword) search.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    Supports optional cross-encoder reranking for improved accuracy.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "local_ai_knowledge",
        llm_host: Optional[str] = None,
        embed_model: Optional[str] = None,
        provider: Optional[str] = None,
        reranker_config: Optional[RerankerConfig] = None,
        enable_reranker: bool = True,
    ):
        settings = get_settings()

        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name

        # Initialize ChromaDB
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize embeddings (supports Ollama, LM Studio, OpenAI-compatible)
        self._embeddings = EmbeddingsProvider(
            host=llm_host or settings.active_host,
            model=embed_model or settings.active_embed_model,
            provider=provider or settings.llm_provider,
        )

        # Initialize BM25 index
        self._bm25 = BM25Index()
        self._sync_bm25_from_chroma()

        # Initialize reranker (lazy-loaded on first use)
        self._enable_reranker = enable_reranker
        self._reranker_config = reranker_config
        self._reranker: Optional[CrossEncoderReranker] = None

        logger.info(
            f"HybridMemory initialized with {self._collection.count()} documents"
        )

    def _sync_bm25_from_chroma(self) -> None:
        """Sync BM25 index from ChromaDB on startup."""
        self._bm25.clear()

        # Get all documents from ChromaDB
        if self._collection.count() == 0:
            return

        results = self._collection.get(include=["documents", "metadatas"])

        documents = []
        for doc_id, content, metadata in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            documents.append(
                Document(content=content, metadata=metadata or {}, id=doc_id)
            )

        self._bm25.add_documents(documents)
        logger.info(f"Synced {len(documents)} documents to BM25 index")

    @property
    def reranker(self) -> Optional[CrossEncoderReranker]:
        """Lazy-load the reranker on first access."""
        if not self._enable_reranker:
            return None

        if self._reranker is None:
            self._reranker = get_reranker(self._reranker_config)
            if self._reranker.is_available():
                logger.info("Cross-encoder reranker initialized")
            else:
                logger.warning("Reranker not available - sentence-transformers may not be installed")

        return self._reranker

    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
        use_diversity: bool = False,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: Initial search results to rerank
            top_k: Number of results to return after reranking
            use_diversity: Whether to use diversity-aware selection

        Returns:
            Reranked search results
        """
        if not results or not self.reranker or not self.reranker.is_available():
            return results[:top_k]

        # Convert SearchResults to dict format for reranker
        docs_for_rerank = [
            {
                "content": r.document.content,
                "metadata": r.document.metadata,
                "doc_id": r.document.id,
                "original_score": r.score,
                "original_source": r.source,
            }
            for r in results
        ]

        # Rerank
        if use_diversity:
            reranked = self.reranker.rerank_with_diversity(query, docs_for_rerank, top_k)
        else:
            reranked = self.reranker.rerank(query, docs_for_rerank, top_k)

        # Convert back to SearchResults
        reranked_results = []
        for doc_dict in reranked:
            doc = Document(
                content=doc_dict["content"],
                metadata=doc_dict["metadata"],
                id=doc_dict["doc_id"],
            )
            reranked_results.append(
                SearchResult(
                    document=doc,
                    score=doc_dict.get("rerank_score", doc_dict.get("original_score", 0)),
                    source=f"{doc_dict['original_source']}+reranked",
                )
            )

        return reranked_results

    def add_document(self, document: Document) -> str:
        """Add a single document to both indexes."""
        return self.add_documents([document])[0]

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add multiple documents to both indexes."""
        if not documents:
            return []

        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self._embeddings.embed_batch(contents)

        # Add to ChromaDB
        ids = [doc.id for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self._collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Add to BM25
        self._bm25.add_documents(documents)

        logger.info(f"Added {len(documents)} documents to memory")
        return ids

    def search_semantic(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search using semantic similarity only."""
        if self._collection.count() == 0:
            return []

        query_embedding = self._embeddings.embed(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert cosine distance to similarity score
            score = 1 - distance
            doc = Document(content=content, metadata=metadata or {}, id=doc_id)
            search_results.append(SearchResult(document=doc, score=score, source="semantic"))

        return search_results

    def search_bm25(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search using BM25 keyword matching only."""
        return self._bm25.search(query, k)

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[SearchResult]:
        """
        Hybrid search combining semantic and BM25 using Reciprocal Rank Fusion.

        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)

        Returns:
            Combined and reranked search results
        """
        # Get results from both methods
        semantic_results = self.search_semantic(query, k=k * 2)
        bm25_results = self.search_bm25(query, k=k * 2)

        # Apply Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # RRF constant (typically 60)
        rrf_k = 60

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + semantic_weight / (
                rrf_k + rank + 1
            )
            doc_map[doc_id] = result.document

        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (
                rrf_k + rank + 1
            )
            doc_map[doc_id] = result.document

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        results = []
        for doc_id in sorted_ids[:k]:
            results.append(
                SearchResult(
                    document=doc_map[doc_id],
                    score=rrf_scores[doc_id],
                    source="hybrid",
                )
            )

        return results

    def search(
        self,
        query: str,
        k: int = 10,
        method: str = "hybrid",
        rerank: bool = False,
        rerank_diversity: bool = False,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Search the memory with optional cross-encoder reranking.

        Args:
            query: Search query
            k: Number of results to return
            method: "hybrid", "semantic", or "bm25"
            rerank: Whether to apply cross-encoder reranking (improves accuracy 20-35%)
            rerank_diversity: Whether to use diversity-aware reranking (avoids redundant results)

        Returns:
            Search results (reranked if enabled)
        """
        # When reranking, retrieve more candidates (10x final k)
        retrieve_k = k * 10 if rerank else k

        if method == "semantic":
            results = self.search_semantic(query, retrieve_k)
        elif method == "bm25":
            results = self.search_bm25(query, retrieve_k)
        else:
            results = self.search_hybrid(query, retrieve_k, **kwargs)

        # Apply reranking if enabled
        if rerank and results:
            results = self._rerank_results(query, results, k, use_diversity=rerank_diversity)
        elif len(results) > k:
            results = results[:k]

        return results

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self._collection.delete(ids=[doc_id])
            # Note: BM25 index would need rebuild for deletion
            # For simplicity, we don't remove from BM25 here
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_documents": self._collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def close(self):
        """Clean up resources."""
        self._embeddings.close()


# Singleton instance
_memory_instance: Optional[HybridMemory] = None


def get_memory() -> HybridMemory:
    """Get or create the memory instance."""
    global _memory_instance
    if _memory_instance is None:
        settings = get_settings()
        _memory_instance = HybridMemory(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
            llm_host=settings.active_host,
            embed_model=settings.active_embed_model,
            provider=settings.llm_provider,
        )
    return _memory_instance
