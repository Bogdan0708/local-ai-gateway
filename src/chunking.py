"""
Advanced chunking strategies for RAG document processing.

Implements:
- Semantic chunking (based on sentence similarity)
- Structure-aware chunking (Markdown/HTML header-based)
- Adaptive chunking (auto-selects strategy based on content)
- Recursive character splitting (fallback)

Research shows:
- Semantic chunking: +70% accuracy vs fixed-size
- Structure-aware: "Single biggest easy improvement"
- Optimal size: 256-512 tokens with 10-20% overlap
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FIXED = "fixed"          # Simple fixed-size chunks
    RECURSIVE = "recursive"  # Recursive character splitting
    SEMANTIC = "semantic"    # Similarity-based boundaries
    STRUCTURE = "structure"  # Header-based for Markdown/HTML
    ADAPTIVE = "adaptive"    # Auto-select based on content


@dataclass
class ChunkConfig:
    """Configuration for chunking."""

    # Strategy to use (ADAPTIVE recommended for production)
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE

    # Target chunk size in characters (not tokens)
    # 400 chars ≈ 100 tokens for English text
    chunk_size: int = 1600  # ~400 tokens

    # Overlap between chunks (10-20% recommended)
    chunk_overlap: int = 200  # ~50 tokens

    # Minimum chunk size (avoid tiny fragments)
    min_chunk_size: int = 100

    # Maximum chunk size (hard limit)
    max_chunk_size: int = 4000  # ~1000 tokens

    # Semantic chunking: similarity threshold for splitting
    # Lower = more splits, higher = fewer splits
    semantic_threshold: float = 0.75

    # Separators for recursive splitting (order matters)
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",    # Paragraph
        "\n",      # Line
        ". ",      # Sentence
        "? ",      # Question
        "! ",      # Exclamation
        "; ",      # Semicolon
        ", ",      # Comma
        " ",       # Word
        "",        # Character
    ])


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    metadata: Dict = field(default_factory=dict)
    index: int = 0

    @property
    def length(self) -> int:
        return len(self.content)


class RecursiveChunker:
    """
    Recursive character text splitter.

    Splits text using hierarchical separators, trying each in order
    until chunks are small enough. This preserves semantic units
    like paragraphs and sentences when possible.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks using recursive splitting."""
        if not text or not text.strip():
            return []

        chunks = self._split_recursive(text, self.config.separators)

        # Add metadata
        return [
            Chunk(content=c, metadata={"strategy": "recursive"}, index=i)
            for i, c in enumerate(chunks)
        ]

    def _split_recursive(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for sep in separators:
            if sep and sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""

                for part in parts:
                    # Would adding this part exceed chunk size?
                    potential = current + sep + part if current else part

                    if len(potential) <= self.config.chunk_size:
                        current = potential
                    else:
                        # Save current chunk if non-empty
                        if current and len(current) >= self.config.min_chunk_size:
                            chunks.append(current)
                        elif current:
                            # Too small, try to merge with next
                            current = potential
                            continue

                        # Start new chunk with overlap
                        if chunks and self.config.chunk_overlap > 0:
                            overlap = chunks[-1][-self.config.chunk_overlap:]
                            current = overlap + part
                        else:
                            current = part

                        # If part itself is too large, split recursively
                        if len(current) > self.config.chunk_size:
                            sub_chunks = self._split_recursive(
                                current, separators[separators.index(sep) + 1:]
                            )
                            chunks.extend(sub_chunks[:-1])
                            current = sub_chunks[-1] if sub_chunks else ""

                # Don't forget the last chunk
                if current and len(current) >= self.config.min_chunk_size:
                    chunks.append(current)
                elif current and chunks:
                    # Merge with previous if too small
                    chunks[-1] += sep + current

                return chunks

        # No separator worked, split by character
        return [
            text[i:i + self.config.chunk_size]
            for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap)
        ]


class SemanticChunker:
    """
    Semantic chunking based on sentence similarity.

    Splits text where semantic meaning changes significantly.
    Requires sentence-transformers for embeddings.

    Performance: +70% accuracy improvement over fixed-size.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self._model = None
        self._model_loaded = False

    @property
    def model(self):
        """Lazy load sentence transformer model."""
        if not self._model_loaded:
            try:
                from sentence_transformers import SentenceTransformer

                # Use a small, fast model for chunking
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._model_loaded = True
                logger.info("Semantic chunking model loaded")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed for semantic chunking. "
                    "Falling back to recursive chunking."
                )
                self._model = None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._model = None

        return self._model

    def chunk(self, text: str) -> List[Chunk]:
        """Split text based on semantic similarity."""
        if not text or not text.strip():
            return []

        # If model not available, fall back to recursive
        if self.model is None:
            logger.debug("Using recursive fallback for semantic chunking")
            fallback = RecursiveChunker(self.config)
            return fallback.chunk(text)

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(content=text, metadata={"strategy": "semantic"}, index=0)]

        # Get embeddings for all sentences
        try:
            import numpy as np

            embeddings = self.model.encode(sentences, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            fallback = RecursiveChunker(self.config)
            return fallback.chunk(text)

        # Find semantic breakpoints
        breakpoints = [0]
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if similarity < self.config.semantic_threshold:
                breakpoints.append(i)
        breakpoints.append(len(sentences))

        # Build chunks from breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk_sentences = sentences[breakpoints[i]:breakpoints[i + 1]]
            chunk_text = " ".join(chunk_sentences)

            # Handle size constraints
            if len(chunk_text) < self.config.min_chunk_size and chunks:
                # Merge with previous
                chunks[-1].content += " " + chunk_text
            elif len(chunk_text) > self.config.max_chunk_size:
                # Split large chunks recursively
                sub_chunker = RecursiveChunker(self.config)
                sub_chunks = sub_chunker.chunk(chunk_text)
                for sc in sub_chunks:
                    sc.metadata["strategy"] = "semantic+recursive"
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={"strategy": "semantic"},
                    index=len(chunks)
                ))

        # Update indices
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Pattern: split on . ! ? followed by space and capital letter
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0


class StructureAwareChunker:
    """
    Structure-aware chunking for Markdown and HTML.

    Splits on headers to preserve document structure.
    Research shows this is the "single biggest easy improvement."
    """

    # Markdown header patterns
    MD_HEADERS = [
        (r'^# (.+)$', 'h1'),
        (r'^## (.+)$', 'h2'),
        (r'^### (.+)$', 'h3'),
        (r'^#### (.+)$', 'h4'),
        (r'^##### (.+)$', 'h5'),
        (r'^###### (.+)$', 'h6'),
    ]

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.recursive = RecursiveChunker(config)

    def chunk_markdown(self, text: str) -> List[Chunk]:
        """Chunk Markdown preserving header hierarchy."""
        if not text or not text.strip():
            return []

        chunks = []
        current_content = []
        current_headers = {}

        for line in text.split('\n'):
            # Check for headers
            header_found = False
            for pattern, level in self.MD_HEADERS:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    # Save previous chunk if exists
                    if current_content:
                        chunk_text = '\n'.join(current_content)
                        if len(chunk_text) >= self.config.min_chunk_size:
                            chunks.append(Chunk(
                                content=chunk_text,
                                metadata={
                                    "strategy": "structure",
                                    "headers": current_headers.copy()
                                },
                                index=len(chunks)
                            ))
                        current_content = []

                    # Update header context
                    current_headers[level] = match.group(1)
                    # Clear lower-level headers
                    levels = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
                    level_idx = levels.index(level)
                    for lower in levels[level_idx + 1:]:
                        current_headers.pop(lower, None)

                    header_found = True
                    break

            current_content.append(line)

        # Don't forget the last chunk
        if current_content:
            chunk_text = '\n'.join(current_content)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        "strategy": "structure",
                        "headers": current_headers.copy()
                    },
                    index=len(chunks)
                ))

        # Split any chunks that are too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) > self.config.max_chunk_size:
                sub_chunks = self.recursive.chunk(chunk.content)
                for sc in sub_chunks:
                    sc.metadata.update(chunk.metadata)
                    sc.metadata["strategy"] = "structure+recursive"
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # Update indices
        for i, chunk in enumerate(final_chunks):
            chunk.index = i

        return final_chunks

    def chunk_html(self, text: str) -> List[Chunk]:
        """Chunk HTML preserving header hierarchy."""
        # Extract text from HTML first
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, 'html.parser')

            chunks = []
            current_content = []
            current_headers = {}

            # Process all elements
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code', 'div']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Save previous chunk
                    if current_content:
                        chunk_text = '\n'.join(current_content)
                        if len(chunk_text) >= self.config.min_chunk_size:
                            chunks.append(Chunk(
                                content=chunk_text,
                                metadata={
                                    "strategy": "structure",
                                    "headers": current_headers.copy()
                                },
                                index=len(chunks)
                            ))
                        current_content = []

                    # Update header
                    current_headers[element.name] = element.get_text().strip()

                else:
                    text = element.get_text().strip()
                    if text:
                        current_content.append(text)

            # Last chunk
            if current_content:
                chunk_text = '\n'.join(current_content)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(Chunk(
                        content=chunk_text,
                        metadata={
                            "strategy": "structure",
                            "headers": current_headers.copy()
                        },
                        index=len(chunks)
                    ))

            return chunks

        except ImportError:
            logger.warning("BeautifulSoup not available, using recursive chunking")
            return self.recursive.chunk(text)


class AdaptiveChunker:
    """
    Adaptive chunker that selects strategy based on content type.

    This is the recommended chunker for production use.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self.semantic = SemanticChunker(config)
        self.structure = StructureAwareChunker(config)
        self.recursive = RecursiveChunker(config)

    def chunk(
        self,
        text: str,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Adaptively chunk content based on type.

        Args:
            text: Content to chunk
            content_type: MIME type (e.g., "text/markdown")
            filename: Original filename for type detection

        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            return []

        # Detect content type from filename if not provided
        detected_type = self._detect_type(content_type, filename, text)

        # Select strategy based on content type
        if detected_type == "markdown":
            logger.debug("Using structure-aware chunking for Markdown")
            return self.structure.chunk_markdown(text)

        elif detected_type == "html":
            logger.debug("Using structure-aware chunking for HTML")
            return self.structure.chunk_html(text)

        elif detected_type == "code":
            logger.debug("Using recursive chunking for code")
            return self.recursive.chunk(text)

        else:
            # Default: try semantic, fall back to recursive
            if self.config.strategy == ChunkingStrategy.SEMANTIC:
                logger.debug("Using semantic chunking")
                return self.semantic.chunk(text)
            else:
                logger.debug("Using recursive chunking")
                return self.recursive.chunk(text)

    def _detect_type(
        self,
        content_type: Optional[str],
        filename: Optional[str],
        text: str,
    ) -> str:
        """Detect content type from various signals."""
        # Check explicit content type
        if content_type:
            if "markdown" in content_type.lower():
                return "markdown"
            if "html" in content_type.lower():
                return "html"

        # Check filename extension
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in (".md", ".markdown"):
                return "markdown"
            if ext in (".html", ".htm"):
                return "html"
            if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp"):
                return "code"

        # Check content patterns
        if text.startswith("<!DOCTYPE") or "<html" in text[:1000]:
            return "html"
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            return "markdown"

        return "text"


# Convenience functions
def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
    **kwargs,
) -> List[Chunk]:
    """
    Chunk text using specified strategy.

    Args:
        text: Text to chunk
        strategy: Chunking strategy to use
        **kwargs: Additional config options

    Returns:
        List of chunks
    """
    config = ChunkConfig(strategy=strategy, **kwargs)

    if strategy == ChunkingStrategy.SEMANTIC:
        chunker = SemanticChunker(config)
    elif strategy == ChunkingStrategy.STRUCTURE:
        chunker = StructureAwareChunker(config)
    elif strategy == ChunkingStrategy.RECURSIVE:
        chunker = RecursiveChunker(config)
    else:
        chunker = AdaptiveChunker(config)

    return chunker.chunk(text)


def chunk_document(
    text: str,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    config: Optional[ChunkConfig] = None,
) -> List[Chunk]:
    """
    Chunk a document with automatic type detection.

    Args:
        text: Document content
        filename: Original filename
        content_type: MIME type
        config: Chunking configuration

    Returns:
        List of chunks with metadata
    """
    chunker = AdaptiveChunker(config)
    return chunker.chunk(text, content_type, filename)


def get_chunker(config: ChunkConfig):
    """
    Get the appropriate chunker based on configuration.

    Args:
        config: Chunking configuration

    Returns:
        Chunker instance
    """
    if config.strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(config)
    elif config.strategy == ChunkingStrategy.STRUCTURE:
        return StructureAwareChunker(config)
    elif config.strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(config)
    else:
        return AdaptiveChunker(config)
