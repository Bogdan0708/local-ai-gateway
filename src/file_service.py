"""
File ingestion service for documents and code.

Implements:
- File watching with watchdog
- Document processing with LangChain loaders
- Intelligent chunking (semantic, structure-aware, adaptive)
- Security controls (whitelist, size limits, path validation)
"""

import fnmatch
import hashlib
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import get_file_whitelist, get_settings
from .memory import Document, get_memory
from .chunking import (
    AdaptiveChunker,
    ChunkConfig,
    ChunkingStrategy,
    SemanticChunker,
    StructureAwareChunker,
    get_chunker,
)

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security check fails."""

    pass


class FileValidator:
    """Validates files against security rules."""

    def __init__(self):
        self.whitelist = get_file_whitelist()
        self.settings = get_settings()

    def validate_path(self, file_path: Path) -> None:
        """
        Validate file path for security issues.

        Raises:
            SecurityError: If validation fails
        """
        # Resolve to absolute path
        resolved = file_path.resolve()

        # Check for path traversal attempts
        path_str = str(resolved)
        if ".." in path_str:
            raise SecurityError(f"Path traversal detected: {file_path}")

        # Check if within allowed directories
        allowed_dirs = [
            self.settings.documents_path.resolve(),
            self.settings.code_path.resolve(),
        ]

        is_allowed = any(
            self._is_subpath(resolved, allowed_dir) for allowed_dir in allowed_dirs
        )
        if not is_allowed:
            raise SecurityError(f"Path outside allowed directories: {file_path}")

    def validate_extension(self, file_path: Path) -> None:
        """Validate file extension against whitelist."""
        suffix = file_path.suffix.lower()
        if suffix not in self.whitelist.allowed_extensions:
            raise SecurityError(f"File extension not allowed: {suffix}")

    def validate_filename(self, file_path: Path) -> None:
        """Check filename against blocked patterns."""
        name = file_path.name.lower()
        path_str = str(file_path).lower()

        for pattern in self.whitelist.blocked_patterns:
            if fnmatch.fnmatch(name, pattern.lower()) or fnmatch.fnmatch(
                path_str, f"*{pattern.lower()}"
            ):
                raise SecurityError(f"File matches blocked pattern: {pattern}")

    def validate_size(self, file_path: Path) -> None:
        """Validate file size."""
        size = file_path.stat().st_size
        if size > self.whitelist.max_file_size_bytes:
            raise SecurityError(
                f"File too large: {size / 1024 / 1024:.1f}MB "
                f"(max: {self.whitelist.max_file_size_bytes / 1024 / 1024}MB)"
            )

    def validate(self, file_path: Path) -> None:
        """Run all validations."""
        self.validate_path(file_path)
        self.validate_extension(file_path)
        self.validate_filename(file_path)
        self.validate_size(file_path)

    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is a subpath of parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


class DocumentProcessor:
    """Processes documents into chunks for indexing."""

    # Language mapping for code files (used for fallback LangChain splitter)
    LANGUAGE_MAP = {
        ".py": Language.PYTHON,
        ".pyi": Language.PYTHON,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".tsx": Language.TS,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".java": Language.JAVA,
        ".c": Language.C,
        ".cpp": Language.CPP,
        ".h": Language.C,
        ".hpp": Language.CPP,
        ".cs": Language.CSHARP,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".kt": Language.KOTLIN,
        ".scala": Language.SCALA,
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
        ".htm": Language.HTML,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
        use_smart_chunking: bool = True,
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking (ADAPTIVE, SEMANTIC, STRUCTURE, etc.)
            use_smart_chunking: Whether to use the new intelligent chunking system
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.use_smart_chunking = use_smart_chunking
        self.validator = FileValidator()

        # Initialize smart chunker config
        self._chunk_config = ChunkConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_file(self, file_path: Path) -> Generator[Document, None, None]:
        """
        Process a file and yield document chunks.

        Args:
            file_path: Path to the file

        Yields:
            Document chunks with metadata
        """
        file_path = Path(file_path)

        # Validate file
        self.validator.validate(file_path)

        # Get file info for metadata
        stat = file_path.stat()
        file_hash = self._compute_hash(file_path)

        base_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "indexed_at": datetime.utcnow().isoformat(),
            "file_hash": file_hash,
        }

        # Load and split document
        try:
            loader = self._get_loader(file_path)
            documents = loader.load()

            # Use smart chunking if enabled and applicable
            if self.use_smart_chunking and self._should_use_smart_chunking(file_path):
                yield from self._process_with_smart_chunking(
                    documents, file_path, file_hash, base_metadata
                )
            else:
                # Fallback to LangChain splitters
                yield from self._process_with_langchain(
                    documents, file_path, file_hash, base_metadata
                )

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise

    def _should_use_smart_chunking(self, file_path: Path) -> bool:
        """Determine if smart chunking should be used for this file type."""
        suffix = file_path.suffix.lower()
        # Use smart chunking for text documents, markdown, HTML
        # Use LangChain for code files (better language-aware splitting)
        smart_extensions = {".txt", ".md", ".markdown", ".html", ".htm", ".json", ".xml", ".csv"}
        return suffix in smart_extensions

    def _process_with_smart_chunking(
        self,
        documents: list,
        file_path: Path,
        file_hash: str,
        base_metadata: dict,
    ) -> Generator[Document, None, None]:
        """Process documents using the intelligent chunking system."""
        # Combine all document content
        full_text = "\n\n".join(doc.page_content for doc in documents)

        # Detect content type
        suffix = file_path.suffix.lower()
        content_type = mimetypes.guess_type(str(file_path))[0] or "text/plain"

        # Get appropriate chunker
        chunker = get_chunker(self._chunk_config)

        # Chunk the content
        chunks = chunker.chunk(
            full_text,
            content_type=content_type,
            filename=file_path.name,
        )

        for i, chunk in enumerate(chunks):
            metadata = {
                **base_metadata,
                "chunk_index": i,
                "chunking_strategy": chunk.metadata.get("strategy", "unknown"),
                "chunk_start": chunk.metadata.get("start_index", 0),
                "chunk_end": chunk.metadata.get("end_index", 0),
            }

            # Add heading context if available (from structure-aware chunking)
            if chunk.metadata.get("heading"):
                metadata["heading"] = chunk.metadata["heading"]
            if chunk.metadata.get("heading_hierarchy"):
                metadata["heading_hierarchy"] = chunk.metadata["heading_hierarchy"]

            yield Document(
                content=chunk.content,
                metadata=metadata,
                id=f"{file_hash}_{i}",
            )

        logger.debug(
            f"Smart chunking: {file_path.name} -> {len(chunks)} chunks "
            f"(strategy: {self.chunking_strategy.value})"
        )

    def _process_with_langchain(
        self,
        documents: list,
        file_path: Path,
        file_hash: str,
        base_metadata: dict,
    ) -> Generator[Document, None, None]:
        """Process documents using LangChain text splitters (for code files)."""
        splitter = self._get_splitter(file_path)

        for i, chunk in enumerate(splitter.split_documents(documents)):
            metadata = {**base_metadata, "chunk_index": i, "chunking_strategy": "langchain"}
            metadata.update(chunk.metadata)

            yield Document(
                content=chunk.page_content,
                metadata=metadata,
                id=f"{file_hash}_{i}",
            )

    def _get_loader(self, file_path: Path):
        """Get the appropriate document loader."""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix in (".md", ".markdown"):
            return UnstructuredMarkdownLoader(str(file_path))
        elif suffix in (".html", ".htm"):
            return BSHTMLLoader(str(file_path))
        else:
            # Default to text loader for code and plain text
            return TextLoader(str(file_path), encoding="utf-8")

    def _get_splitter(self, file_path: Path):
        """Get the appropriate text splitter."""
        suffix = file_path.suffix.lower()

        if suffix in self.LANGUAGE_MAP:
            return RecursiveCharacterTextSplitter.from_language(
                language=self.LANGUAGE_MAP[suffix],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]


class FileWatcherHandler(FileSystemEventHandler):
    """Handle file system events for auto-indexing."""

    def __init__(self, processor: DocumentProcessor, memory):
        self.processor = processor
        self.memory = memory
        self._processed_hashes: set[str] = set()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._process_file(Path(event.src_path))

    def _process_file(self, file_path: Path) -> None:
        """Process a file and add to memory."""
        try:
            documents = list(self.processor.process_file(file_path))
            if documents:
                self.memory.add_documents(documents)
                logger.info(f"Indexed {len(documents)} chunks from {file_path}")
        except SecurityError as e:
            logger.debug(f"Skipped {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")


class FileIngestionService:
    """Service for ingesting files into the knowledge base."""

    def __init__(
        self,
        watch: bool = False,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE,
        use_smart_chunking: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize file ingestion service.

        Args:
            watch: Whether to watch directories for changes
            chunking_strategy: Strategy for chunking documents
            use_smart_chunking: Use intelligent chunking (semantic/structure-aware)
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.settings = get_settings()
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            use_smart_chunking=use_smart_chunking,
        )
        self.memory = get_memory()
        self._observer: Optional[Observer] = None
        self._watching = watch

        if watch:
            self._start_watching()

    def ingest_file(self, file_path: Path) -> int:
        """
        Ingest a single file.

        Returns:
            Number of chunks indexed
        """
        documents = list(self.processor.process_file(file_path))
        if documents:
            self.memory.add_documents(documents)
        return len(documents)

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> dict[str, int]:
        """
        Ingest all files in a directory.

        Returns:
            Dict mapping file paths to chunk counts
        """
        results = {}
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    count = self.ingest_file(file_path)
                    results[str(file_path)] = count
                except SecurityError as e:
                    logger.debug(f"Skipped {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    results[str(file_path)] = 0

        logger.info(
            f"Ingested {len(results)} files with "
            f"{sum(results.values())} total chunks"
        )
        return results

    def _start_watching(self) -> None:
        """Start file system watcher."""
        handler = FileWatcherHandler(self.processor, self.memory)
        self._observer = Observer()

        # Watch documents directory
        if self.settings.documents_path.exists():
            self._observer.schedule(
                handler,
                str(self.settings.documents_path),
                recursive=True,
            )
            logger.info(f"Watching {self.settings.documents_path}")

        # Watch code directory
        if self.settings.code_path.exists():
            self._observer.schedule(
                handler,
                str(self.settings.code_path),
                recursive=True,
            )
            logger.info(f"Watching {self.settings.code_path}")

        self._observer.start()

    def stop_watching(self) -> None:
        """Stop file system watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "memory_stats": self.memory.get_stats(),
            "watching": self._watching,
            "documents_path": str(self.settings.documents_path),
            "code_path": str(self.settings.code_path),
        }


# Convenience functions
def ingest_file(file_path: Path) -> int:
    """Ingest a single file."""
    service = FileIngestionService()
    return service.ingest_file(file_path)


def ingest_directory(directory: Path, recursive: bool = True) -> dict[str, int]:
    """Ingest all files in a directory."""
    service = FileIngestionService()
    return service.ingest_directory(directory, recursive)
