"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Security ===
    jwt_secret: str = Field(..., description="Secret key for JWT signing")
    api_key: str = Field(..., description="API key for authentication")
    encryption_key: Optional[str] = Field(None, description="Key for encrypting sensitive data")

    # === LLM Provider ===
    llm_provider: str = Field(default="lmstudio", description="ollama, lmstudio, or openai-compatible")

    # LM Studio (OpenAI-compatible)
    lmstudio_host: str = Field(default="http://localhost:1234")
    lmstudio_chat_model: str = Field(default="gpt-oss-120b")
    lmstudio_embed_model: str = Field(default="nomic-embed-text")

    # Ollama
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="llama3.1")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # OpenAI-compatible generic
    openai_base_url: str = Field(default="http://localhost:1234/v1")
    openai_api_key: str = Field(default="lm-studio")

    @property
    def active_chat_model(self) -> str:
        """Get active chat model based on provider."""
        if self.llm_provider == "lmstudio":
            return self.lmstudio_chat_model
        elif self.llm_provider == "ollama":
            return self.ollama_chat_model
        return self.lmstudio_chat_model

    @property
    def active_embed_model(self) -> str:
        """Get active embedding model based on provider."""
        if self.llm_provider == "lmstudio":
            return self.lmstudio_embed_model
        elif self.llm_provider == "ollama":
            return self.ollama_embed_model
        return self.lmstudio_embed_model

    @property
    def active_host(self) -> str:
        """Get active LLM host based on provider."""
        if self.llm_provider == "lmstudio":
            return self.lmstudio_host
        elif self.llm_provider == "ollama":
            return self.ollama_host
        return self.openai_base_url.rstrip("/v1")

    # === Server ===
    # When true, requests arriving on the loopback interface may skip
    # authentication. Off by default: the server binds 0.0.0.0 out of the box,
    # and a proxy can make remote traffic appear to come from 127.0.0.1.
    allow_loopback_unauthenticated: bool = Field(
        default=False,
        description="Allow unauthenticated access from 127.0.0.1/::1 only",
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    cors_allow_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of origins allowed to call the API",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list, empty entries dropped."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def binds_loopback_only(self) -> bool:
        """True when the server listens on loopback only."""
        return self.host in ("127.0.0.1", "::1", "localhost")

    # === Rate Limiting ===
    rate_limit_requests: int = Field(default=100, description="Max requests per period")
    rate_limit_period: int = Field(default=3600, description="Rate limit period in seconds")

    # === File Ingestion ===
    max_file_size_mb: int = Field(default=50)
    max_total_storage_gb: int = Field(default=10)
    documents_path: Path = Field(default=Path("/data/documents"))
    code_path: Path = Field(default=Path("/data/code"))

    # === Web Fetcher ===
    # Disable allowlist enforcement for outbound fetches. The blocklist and
    # every SSRF/IP check still apply; only the domain allowlist is skipped.
    allow_all_domains: bool = Field(
        default=False,
        description="Skip the domain allowlist (fail-open) for web fetches",
    )
    # Tier 3 uses a real browser, which resolves and navigates on its own.
    # Off by default: callers cannot ask for it unless the operator enables it.
    allow_browser_tier: bool = Field(
        default=False,
        description="Allow the Playwright (browser) fetch tier",
    )
    web_fetch_timeout: int = Field(default=30)
    web_fetch_max_size_mb: int = Field(default=5)
    web_fetch_rate_limit: int = Field(default=10, description="Requests per minute")

    # === ChromaDB ===
    chroma_persist_dir: Path = Field(default=Path("/data/chroma"))
    chroma_collection_name: str = Field(default="local_ai_knowledge")

    # === Paths ===
    config_dir: Path = Field(default=Path("config"))
    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("data/logs"))

    @field_validator("jwt_secret", "api_key")
    @classmethod
    def validate_secrets(cls, v: str) -> str:
        """Ensure secrets are not placeholder values."""
        if not v or v.startswith("your-") or len(v) < 32:
            raise ValueError(
                "Secret must be at least 32 characters. "
                "Generate with: openssl rand -hex 32"
            )
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_total_storage_bytes(self) -> int:
        return self.max_total_storage_gb * 1024 * 1024 * 1024

    @property
    def web_fetch_max_size_bytes(self) -> int:
        return self.web_fetch_max_size_mb * 1024 * 1024


# Mirrors config/allowed_domains.example.yaml, for the case where neither the
# operator's file nor the example is on disk. The allowlist is enforced
# fail-closed, so "no configuration" must not mean "fetch anything".
DEFAULT_ALLOWED_DOMAINS = [
    "*.wikipedia.org",
    "*.python.org",
    "developer.mozilla.org",
    "*.readthedocs.io",
    "pypi.org",
    "github.com",
    "*.github.com",
    "raw.githubusercontent.com",
    "stackoverflow.com",
    "*.arxiv.org",
    "arxiv.org",
]
DEFAULT_BLOCKED_DOMAINS = [
    "*.local",
    "*.internal",
    "*.localdomain",
    "metadata.google.internal",
]


def _example_path(config_path: Path) -> Path:
    """Sibling ``<name>.example<suffix>`` path for a config file."""
    return config_path.with_name(
        f"{config_path.stem}.example{config_path.suffix}"
    )


class DomainConfig:
    """Load and manage domain whitelist/blocklist configuration."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._allowed: list[str] = []
        self._blocked: list[str] = []
        self._load()

    def _load(self) -> None:
        """
        Load configuration from YAML.

        Resolution order: the operator's file, then the shipped example, then
        the in-code defaults - the same order as the file whitelist.
        """
        source = None
        if self.config_path.exists():
            source = self.config_path
        else:
            example = _example_path(self.config_path)
            if example.exists():
                source = example

        data: dict = {}
        if source is not None:
            with open(source) as f:
                data = yaml.safe_load(f) or {}
            self.config_path = source

        self._allowed = data.get("allowed", list(DEFAULT_ALLOWED_DOMAINS))
        self._blocked = data.get("blocked", list(DEFAULT_BLOCKED_DOMAINS))

    @property
    def allowed(self) -> list[str]:
        return self._allowed

    @property
    def blocked(self) -> list[str]:
        return self._blocked


# Safe defaults used when no whitelist file is present on disk. They mirror
# config/file_whitelist.example.yaml so a clean checkout is usable but still
# conservative.
DEFAULT_ALLOWED_EXTENSIONS = [".txt", ".md", ".pdf", ".csv", ".json", ".yaml"]
DEFAULT_BLOCKED_PATTERNS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_rsa*",
    "id_ed25519*",
    "*.kdbx",
    "credentials",
    "credentials.*",
    ".git/*",
    ".ssh/*",
    "*.sqlite",
    "*.db",
]


class FileWhitelistConfig:
    """Load and manage file extension whitelist configuration."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._allowed_extensions: list[str] = []
        self._blocked_patterns: list[str] = []
        self._max_file_size_mb: int = 50
        self._max_total_storage_gb: int = 10
        self._load()

    def _load(self) -> None:
        """
        Load configuration from YAML.

        Resolution order: the operator's file, then the shipped example file,
        then the in-code defaults. An absent file must never mean "allow
        nothing" -- that silently breaks ingestion on a clean checkout.
        """
        source = None
        if self.config_path.exists():
            source = self.config_path
        else:
            example = _example_path(self.config_path)
            if example.exists():
                source = example

        data: dict = {}
        if source is not None:
            with open(source) as f:
                data = yaml.safe_load(f) or {}
            self.config_path = source

        self._allowed_extensions = data.get(
            "allowed_extensions", list(DEFAULT_ALLOWED_EXTENSIONS)
        )
        self._blocked_patterns = data.get(
            "blocked_patterns", list(DEFAULT_BLOCKED_PATTERNS)
        )
        self._max_file_size_mb = data.get("max_file_size_mb", 50)
        self._max_total_storage_gb = data.get("max_total_storage_gb", 10)

    @property
    def allowed_extensions(self) -> list[str]:
        return self._allowed_extensions

    @property
    def blocked_patterns(self) -> list[str]:
        return self._blocked_patterns

    @property
    def max_file_size_bytes(self) -> int:
        return self._max_file_size_mb * 1024 * 1024

    @property
    def max_total_storage_bytes(self) -> int:
        return self._max_total_storage_gb * 1024 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache
def get_domain_config() -> DomainConfig:
    """Get cached domain configuration."""
    settings = get_settings()
    return DomainConfig(settings.config_dir / "allowed_domains.yaml")


@lru_cache
def get_file_whitelist() -> FileWhitelistConfig:
    """Get cached file whitelist configuration."""
    settings = get_settings()
    return FileWhitelistConfig(settings.config_dir / "file_whitelist.yaml")
