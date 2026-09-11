"""
Cross-encoder reranking for improved RAG retrieval quality.

Implements:
- Cross-encoder based reranking using sentence-transformers
- Lazy model loading for efficiency
- Configurable top-k selection
- Optional diversity-aware reranking

Performance impact:
- Typical improvement: +20-35% accuracy
- Additional latency: 200-500ms for 50 candidates
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for reranking."""

    # Model to use for reranking
    # Options: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, general)
    #          BAAI/bge-reranker-base (multilingual)
    #          cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate)
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Batch size for scoring
    batch_size: int = 32

    # Number of results to return after reranking
    top_k: int = 5

    # Number of candidates to retrieve for reranking
    # Recommend: 10x final top_k for good coverage
    retrieve_k: int = 50

    # Enable diversity-aware selection (MMR-style)
    use_diversity: bool = False

    # Diversity threshold (lower = more diverse)
    diversity_threshold: float = 0.7


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval quality.

    Cross-encoders process query and document together (concatenated),
    providing more accurate relevance scores than bi-encoders which
    encode query and document separately.

    Usage:
        reranker = CrossEncoderReranker()
        results = reranker.rerank("What is Python?", documents)
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize reranker with optional configuration.

        Args:
            config: Reranker configuration. Uses defaults if not provided.
        """
        self.config = config or RerankerConfig()
        self._model = None
        self._model_loaded = False

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if not self._model_loaded:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading reranker model: {self.config.model_name}")
                self._model = CrossEncoder(
                    self.config.model_name,
                    max_length=512,
                )
                self._model_loaded = True
                logger.info("Reranker model loaded successfully")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                self._model = None
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                self._model = None

        return self._model

    def is_available(self) -> bool:
        """Check if reranker is available (model can be loaded)."""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of dicts with at least 'content' key
            top_k: Number of results to return (default: config.top_k)

        Returns:
            Reranked documents with 'rerank_score' added
        """
        if not documents:
            return []

        if not self.is_available():
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k or self.config.top_k]

        top_k = top_k or self.config.top_k

        # Prepare query-document pairs for scoring
        pairs = [
            (query, doc.get("content", doc.get("text", "")))
            for doc in documents
        ]

        # Score with cross-encoder
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0.0)

        # Sort by rerank score (descending)
        reranked = sorted(
            documents,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )

        logger.debug(
            f"Reranked {len(documents)} documents, "
            f"top score: {reranked[0].get('rerank_score', 0):.4f}"
        )

        return reranked[:top_k]

    def rerank_with_diversity(
        self,
        query: str,
        documents: List[dict],
        top_k: Optional[int] = None,
        diversity_threshold: Optional[float] = None,
    ) -> List[dict]:
        """
        Rerank with diversity to avoid redundant results.

        Uses a simple MMR-style selection that avoids selecting
        documents too similar to already selected ones.

        Args:
            query: Search query
            documents: List of document dicts
            top_k: Number of results to return
            diversity_threshold: Similarity threshold (0-1, lower = more diverse)

        Returns:
            Diverse set of reranked documents
        """
        if not documents:
            return []

        if not self.is_available():
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k or self.config.top_k]

        top_k = top_k or self.config.top_k
        threshold = diversity_threshold or self.config.diversity_threshold

        # First, get rerank scores
        pairs = [
            (query, doc.get("content", doc.get("text", "")))
            for doc in documents
        ]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by score
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )

        # Select diverse results using simple overlap check
        selected = [sorted_docs[0]] if sorted_docs else []

        for doc in sorted_docs[1:]:
            if len(selected) >= top_k:
                break

            doc_content = doc.get("content", doc.get("text", ""))
            doc_words = set(doc_content.lower().split())

            # Check similarity with already selected documents
            is_diverse = True
            for sel in selected:
                sel_content = sel.get("content", sel.get("text", ""))
                sel_words = set(sel_content.lower().split())

                # Calculate Jaccard similarity
                if doc_words and sel_words:
                    intersection = len(doc_words & sel_words)
                    union = len(doc_words | sel_words)
                    similarity = intersection / union if union > 0 else 0

                    if similarity > threshold:
                        is_diverse = False
                        break

            if is_diverse:
                selected.append(doc)

        logger.debug(
            f"Selected {len(selected)} diverse documents from {len(documents)}"
        )

        return selected


# Singleton instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(config: Optional[RerankerConfig] = None) -> CrossEncoderReranker:
    """Get or create the reranker singleton instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(config)
    return _reranker


def rerank_documents(
    query: str,
    documents: List[dict],
    top_k: int = 5,
    use_diversity: bool = False,
) -> List[dict]:
    """
    Convenience function to rerank documents.

    Args:
        query: Search query
        documents: List of document dicts with 'content' key
        top_k: Number of results to return
        use_diversity: Whether to use diversity-aware selection

    Returns:
        Reranked documents
    """
    reranker = get_reranker()

    if use_diversity:
        return reranker.rerank_with_diversity(query, documents, top_k)
    else:
        return reranker.rerank(query, documents, top_k)
