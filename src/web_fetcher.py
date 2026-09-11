"""
Web fetcher with SSRF protection and multi-tier fallback.

Implements:
- SSRF protection (blocks private IPs, localhost, metadata servers)
- DNS rebinding protection (resolve once, validate, cache)
- Domain whitelist/blocklist
- Multi-tier fallback (httpx -> curl -> playwright)
- Rate limiting
- Content sanitization

Security improvements (2025-12-31):
- Fixed DNS rebinding vulnerability by resolving DNS once and caching
- Added IPv6 support using getaddrinfo
- Added IP caching with configurable TTL
"""

import fnmatch
import hashlib
import ipaddress
import logging
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from .config import get_domain_config, get_settings

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security check fails."""

    pass


class FetchError(Exception):
    """Raised when fetching fails."""

    pass


@dataclass
class FetchResult:
    """Result of a web fetch operation."""

    url: str
    content: str
    content_type: str
    status_code: int
    tier_used: str  # "httpx", "curl", or "playwright"
    fetched_at: str
    metadata: dict


# SSRF Protection - Block these networks
BLOCKED_IP_NETWORKS = [
    # IPv4 blocked ranges
    ipaddress.ip_network('127.0.0.0/8'),       # Loopback
    ipaddress.ip_network('0.0.0.0/8'),         # This network
    ipaddress.ip_network('10.0.0.0/8'),        # Private (Class A)
    ipaddress.ip_network('172.16.0.0/12'),     # Private (Class B)
    ipaddress.ip_network('192.168.0.0/16'),    # Private (Class C)
    ipaddress.ip_network('169.254.0.0/16'),    # Link-local / AWS metadata
    ipaddress.ip_network('224.0.0.0/4'),       # Multicast
    ipaddress.ip_network('240.0.0.0/4'),       # Reserved
    # IPv6 blocked ranges
    ipaddress.ip_network('::1/128'),           # Loopback
    ipaddress.ip_network('::/128'),            # Unspecified
    ipaddress.ip_network('fc00::/7'),          # Unique local
    ipaddress.ip_network('fe80::/10'),         # Link-local
    ipaddress.ip_network('ff00::/8'),          # Multicast
]

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.internal",
    "169.254.169.254",  # AWS/cloud metadata as hostname
}


class SafeDNSResolver:
    """
    DNS resolver with rebinding protection.

    Resolves DNS once, validates ALL resolved IPs, and caches results.
    This prevents DNS rebinding attacks where an attacker's DNS server
    returns a safe IP for validation but a malicious IP for the actual request.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize resolver with caching.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: dict[str, Tuple[List[str], float]] = {}

    def resolve_and_validate(self, url: str) -> Tuple[str, str]:
        """
        Resolve DNS once, validate all IPs, return safe IP.

        This is the core of DNS rebinding protection:
        1. Parse hostname from URL
        2. Check hostname blocklist
        3. Resolve ALL IP addresses (IPv4 and IPv6)
        4. Validate ALL resolved IPs against blocked ranges
        5. Cache the result
        6. Return the first valid IP

        Args:
            url: URL to resolve

        Returns:
            Tuple of (hostname, validated_ip)

        Raises:
            SecurityError: If hostname or any resolved IP is blocked
        """
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            raise SecurityError("No hostname in URL")

        # Check hostname blocklist
        if hostname.lower() in BLOCKED_HOSTNAMES:
            raise SecurityError(f"Blocked hostname: {hostname}")

        # Check cache first
        cached_ips = self._get_cached(hostname)
        if cached_ips:
            logger.debug(f"DNS cache hit: {hostname} -> {cached_ips[0]}")
            return hostname, cached_ips[0]

        # Resolve ALL addresses (both IPv4 and IPv6)
        try:
            addr_infos = socket.getaddrinfo(
                hostname,
                None,
                socket.AF_UNSPEC,  # Both IPv4 and IPv6
                socket.SOCK_STREAM
            )
        except socket.gaierror as e:
            raise SecurityError(f"DNS resolution failed for {hostname}: {e}")

        if not addr_infos:
            raise SecurityError(f"No DNS records found for {hostname}")

        # Extract unique IPs
        ips = list(set(info[4][0] for info in addr_infos))

        # Validate ALL resolved IPs - if any is blocked, reject
        for ip in ips:
            self._validate_ip(ip, hostname)

        # Cache the validated IPs
        self._cache[hostname] = (ips, time.time())

        logger.debug(f"DNS resolved and validated: {hostname} -> {ips[0]}")
        return hostname, ips[0]

    def _validate_ip(self, ip_str: str, hostname: str) -> None:
        """
        Validate an IP address is not in blocked ranges.

        Args:
            ip_str: IP address string
            hostname: Original hostname (for error messages)

        Raises:
            SecurityError: If IP is in a blocked range
        """
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SecurityError(f"Invalid IP address: {ip_str}")

        for blocked_net in BLOCKED_IP_NETWORKS:
            if ip_obj in blocked_net:
                raise SecurityError(
                    f"SSRF blocked: {hostname} resolves to {ip_str} "
                    f"(blocked network: {blocked_net})"
                )

    def _get_cached(self, hostname: str) -> Optional[List[str]]:
        """Get cached IPs if not expired."""
        if hostname in self._cache:
            ips, timestamp = self._cache[hostname]
            if time.time() - timestamp < self.cache_ttl:
                return ips
            # Expired - remove from cache
            del self._cache[hostname]
        return None

    def clear_cache(self) -> None:
        """Clear the DNS cache."""
        self._cache.clear()

    def build_url_with_ip(self, url: str, ip: str) -> Tuple[str, str]:
        """
        Build URL with IP instead of hostname for the actual request.

        Returns the modified URL and the original Host header value.
        This ensures the request goes to the validated IP while
        preserving the Host header for virtual hosting.

        Args:
            url: Original URL
            ip: Validated IP address

        Returns:
            Tuple of (url_with_ip, original_hostname)
        """
        parsed = urlparse(url)
        original_hostname = parsed.hostname

        # Replace hostname with IP in netloc
        if parsed.port:
            new_netloc = f"{ip}:{parsed.port}"
        else:
            new_netloc = ip

        # Rebuild URL with IP
        url_with_ip = urlunparse((
            parsed.scheme,
            new_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        return url_with_ip, original_hostname


class SSRFValidator:
    """
    Validates URLs against SSRF attacks with DNS rebinding protection.

    Uses SafeDNSResolver to ensure DNS is resolved only once and
    all resolved IPs are validated before any request is made.
    """

    def __init__(self):
        self.dns_resolver = SafeDNSResolver(cache_ttl=300)
        self.blocked_hostnames = BLOCKED_HOSTNAMES

    def validate(self, url: str) -> Tuple[str, str]:
        """
        Validate URL is not an SSRF target.

        Returns validated IP to use for the actual request,
        preventing DNS rebinding attacks.

        Args:
            url: URL to validate

        Returns:
            Tuple of (hostname, validated_ip) to use for the request

        Raises:
            SecurityError: If URL is blocked
        """
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise SecurityError(f"Invalid scheme: {parsed.scheme}")

        # Delegate to DNS resolver (handles hostname check + IP validation)
        hostname, validated_ip = self.dns_resolver.resolve_and_validate(url)

        logger.debug(f"SSRF validation passed: {hostname} -> {validated_ip}")
        return hostname, validated_ip

    def clear_dns_cache(self) -> None:
        """Clear the DNS resolution cache."""
        self.dns_resolver.clear_cache()


class DomainValidator:
    """Validates URLs against domain whitelist/blocklist."""

    def __init__(self):
        self.config = get_domain_config()

    def validate(self, url: str) -> None:
        """
        Validate URL against domain configuration.

        Raises:
            SecurityError: If domain is not allowed
        """
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            raise SecurityError("No hostname in URL")

        # Check blocklist first (takes precedence)
        for pattern in self.config.blocked:
            if self._matches(hostname, pattern):
                raise SecurityError(f"Domain blocked: {hostname}")

        # Check whitelist
        if self.config.allowed:
            allowed = any(
                self._matches(hostname, pattern) for pattern in self.config.allowed
            )
            if not allowed:
                raise SecurityError(f"Domain not in whitelist: {hostname}")

    def _matches(self, hostname: str, pattern: str) -> bool:
        """Check if hostname matches pattern (supports wildcards)."""
        # Handle wildcard patterns like *.example.com
        if pattern.startswith("*."):
            # Match the domain itself or any subdomain
            domain = pattern[2:]
            return hostname == domain or hostname.endswith(f".{domain}")
        else:
            return hostname == pattern


class ContentCache:
    """Simple in-memory cache for fetched content."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[FetchResult, datetime]] = {}

    def get(self, url: str) -> Optional[FetchResult]:
        """Get cached result if not expired."""
        key = self._make_key(url)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.utcnow() - timestamp < self.ttl:
                return result
            else:
                del self._cache[key]
        return None

    def set(self, url: str, result: FetchResult) -> None:
        """Cache a result."""
        key = self._make_key(url)
        self._cache[key] = (result, datetime.utcnow())

    def _make_key(self, url: str) -> str:
        """Create cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class ContentSanitizer:
    """Sanitizes fetched HTML content."""

    # Tags to remove completely
    REMOVE_TAGS = [
        "script",
        "style",
        "iframe",
        "object",
        "embed",
        "form",
        "input",
        "button",
        "noscript",
        "svg",
        "canvas",
    ]

    # Attributes to remove
    REMOVE_ATTRS = [
        "onclick",
        "onload",
        "onerror",
        "onmouseover",
        "onfocus",
        "onblur",
        "onsubmit",
        "style",
    ]

    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML and convert to clean text/markdown."""
        soup = BeautifulSoup(html, "lxml")

        # Remove unwanted tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove unwanted attributes
        for tag in soup.find_all(True):
            for attr in self.REMOVE_ATTRS:
                if attr in tag.attrs:
                    del tag.attrs[attr]

        # Extract text with some structure
        return self._html_to_markdown(soup)

    def _html_to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert HTML to simple markdown."""
        lines = []

        # Get title
        title = soup.find("title")
        if title:
            lines.append(f"# {title.get_text().strip()}\n")

        # Get main content (prefer article, main, or body)
        content = soup.find("article") or soup.find("main") or soup.find("body")
        if not content:
            content = soup

        # Process content
        for element in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            text = element.get_text().strip()
            if not text:
                continue

            if element.name == "h1":
                lines.append(f"\n# {text}\n")
            elif element.name == "h2":
                lines.append(f"\n## {text}\n")
            elif element.name == "h3":
                lines.append(f"\n### {text}\n")
            elif element.name == "h4":
                lines.append(f"\n#### {text}\n")
            elif element.name == "li":
                lines.append(f"- {text}")
            elif element.name in ("pre", "code"):
                lines.append(f"\n```\n{text}\n```\n")
            else:
                lines.append(f"{text}\n")

        return "\n".join(lines)


class WebFetcher:
    """
    Multi-tier web fetcher with security protections.

    Tiers:
    1. httpx - Fast, simple HTTP client
    2. curl - With Chrome headers for basic bot bypass
    3. playwright - Full browser for JS-rendered content
    """

    # Chrome-like headers for curl tier
    CHROME_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    def __init__(self):
        self.settings = get_settings()
        self.ssrf_validator = SSRFValidator()
        self.domain_validator = DomainValidator()
        self.sanitizer = ContentSanitizer()
        self.cache = ContentCache(ttl_seconds=3600)

        self._http_client = httpx.Client(
            timeout=self.settings.web_fetch_timeout,
            follow_redirects=True,
            max_redirects=5,
        )

    def fetch(
        self,
        url: str,
        use_cache: bool = True,
        max_tier: int = 2,  # 1=httpx, 2=curl, 3=playwright
    ) -> FetchResult:
        """
        Fetch URL content with multi-tier fallback.

        Args:
            url: URL to fetch
            use_cache: Whether to use cached results
            max_tier: Maximum tier to try (1-3)

        Returns:
            FetchResult with content and metadata

        Raises:
            SecurityError: If URL fails security checks
            FetchError: If all tiers fail
        """
        # Security validations FIRST - returns validated IP to use
        hostname, validated_ip = self.ssrf_validator.validate(url)
        self.domain_validator.validate(url)

        # Build URL with validated IP to prevent DNS rebinding
        safe_url, original_host = self.ssrf_validator.dns_resolver.build_url_with_ip(
            url, validated_ip
        )

        # Check cache
        if use_cache:
            cached = self.cache.get(url)
            if cached:
                logger.debug(f"Cache hit for {url}")
                return cached

        # Try tiers in order (using safe_url with validated IP)
        errors = []

        # Tier 1: httpx
        if max_tier >= 1:
            try:
                result = self._fetch_httpx(safe_url, original_host, url)
                if use_cache:
                    self.cache.set(url, result)
                return result
            except Exception as e:
                errors.append(f"httpx: {e}")
                logger.debug(f"Tier 1 (httpx) failed for {url}: {e}")

        # Tier 2: curl with Chrome headers
        if max_tier >= 2:
            try:
                result = self._fetch_curl(safe_url, original_host, url)
                if use_cache:
                    self.cache.set(url, result)
                return result
            except Exception as e:
                errors.append(f"curl: {e}")
                logger.debug(f"Tier 2 (curl) failed for {url}: {e}")

        # Tier 3: playwright (if available and enabled)
        # Note: Playwright uses original URL as it handles DNS internally
        # but we've already validated the hostname resolves to a safe IP
        if max_tier >= 3:
            try:
                result = self._fetch_playwright(url)
                if use_cache:
                    self.cache.set(url, result)
                return result
            except Exception as e:
                errors.append(f"playwright: {e}")
                logger.debug(f"Tier 3 (playwright) failed for {url}: {e}")

        raise FetchError(f"All tiers failed: {'; '.join(errors)}")

    def _fetch_httpx(
        self, safe_url: str, original_host: str, original_url: str
    ) -> FetchResult:
        """
        Tier 1: Simple HTTP fetch with DNS rebinding protection.

        Args:
            safe_url: URL with validated IP instead of hostname
            original_host: Original hostname for Host header
            original_url: Original URL for result reporting
        """
        # Set Host header to original hostname (required for virtual hosting)
        headers = {"Host": original_host}

        response = self._http_client.get(safe_url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "text/html")
        content = response.text

        # Sanitize if HTML
        if "html" in content_type.lower():
            content = self.sanitizer.sanitize_html(content)

        # Check size limit
        if len(content) > self.settings.web_fetch_max_size_bytes:
            content = content[: self.settings.web_fetch_max_size_bytes]
            logger.warning(f"Content truncated for {original_url}")

        return FetchResult(
            url=original_url,  # Return original URL, not the IP-based one
            content=content,
            content_type=content_type,
            status_code=response.status_code,
            tier_used="httpx",
            fetched_at=datetime.utcnow().isoformat(),
            metadata={
                "headers": dict(response.headers),
            },
        )

    def _fetch_curl(
        self, safe_url: str, original_host: str, original_url: str
    ) -> FetchResult:
        """
        Tier 2: curl with Chrome-like headers and DNS rebinding protection.

        Args:
            safe_url: URL with validated IP instead of hostname
            original_host: Original hostname for Host header
            original_url: Original URL for result reporting
        """
        # Build curl command with headers
        cmd = ["curl", "-sL", "--compressed", "-m", str(self.settings.web_fetch_timeout)]

        # Add Host header for virtual hosting
        cmd.extend(["-H", f"Host: {original_host}"])

        for key, value in self.CHROME_HEADERS.items():
            cmd.extend(["-H", f"{key}: {value}"])

        cmd.append(safe_url)

        # Execute curl (using subprocess with list args - no shell injection)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.settings.web_fetch_timeout + 5,
        )

        if result.returncode != 0:
            raise FetchError(f"curl failed: {result.stderr}")

        content = result.stdout

        # Sanitize HTML
        content = self.sanitizer.sanitize_html(content)

        # Check size limit
        if len(content) > self.settings.web_fetch_max_size_bytes:
            content = content[: self.settings.web_fetch_max_size_bytes]

        return FetchResult(
            url=original_url,  # Return original URL
            content=content,
            content_type="text/html",
            status_code=200,
            tier_used="curl",
            fetched_at=datetime.utcnow().isoformat(),
            metadata={},
        )

    def _fetch_playwright(self, url: str) -> FetchResult:
        """Tier 3: Full browser rendering with Playwright."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise FetchError("Playwright not installed")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_default_timeout(self.settings.web_fetch_timeout * 1000)

                response = page.goto(url, wait_until="networkidle")

                if not response or response.status >= 400:
                    raise FetchError(f"Page load failed: {response.status if response else 'no response'}")

                content = page.content()
                content = self.sanitizer.sanitize_html(content)

                # Check size limit
                if len(content) > self.settings.web_fetch_max_size_bytes:
                    content = content[: self.settings.web_fetch_max_size_bytes]

                return FetchResult(
                    url=page.url,
                    content=content,
                    content_type="text/html",
                    status_code=response.status,
                    tier_used="playwright",
                    fetched_at=datetime.utcnow().isoformat(),
                    metadata={},
                )
            finally:
                browser.close()

    def close(self):
        """Clean up resources."""
        self._http_client.close()


# Singleton instance
_fetcher_instance: Optional[WebFetcher] = None


def get_web_fetcher() -> WebFetcher:
    """Get or create the web fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = WebFetcher()
    return _fetcher_instance


def fetch_url(
    url: str,
    use_cache: bool = True,
    max_tier: int = 2,
    timeout: int = 30,
) -> FetchResult:
    """
    Fetch content from a URL with multi-tier fallback.

    Tiers:
    1. HTTPX (Standard HTTP)
    2. Curl (Impersonate Browser)
    3. Playwright (Headless Browser - JS support)
    """
    # Check cache
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            # Check age (default 24h)
            fetched_at = datetime.fromisoformat(data["fetched_at"])
            if datetime.utcnow() - fetched_at < timedelta(hours=24):
                logger.info(f"Cache hit for {url}")
                return FetchResult(**data)
        except Exception as e:
            logger.warning(f"Cache read failed for {url}: {e}")

    # Validate URL and SSRF protection
    _validate_url(url)

    error = None
    result = None

    # Tier 1: HTTPX
    if max_tier >= 1:
        try:
            result = _fetch_httpx(url, timeout)
        except Exception as e:
            error = e
            logger.debug(f"Tier 1 failed for {url}: {e}")

    # Tier 2: Curl (via subprocess)
    if not result and max_tier >= 2:
        try:
            result = _fetch_curl(url, timeout)
        except Exception as e:
            error = e
            logger.debug(f"Tier 2 failed for {url}: {e}")

    # Tier 3: Playwright (Not implemented in this snippet to keep deps light)
    # Could be added if needed for JS-heavy sites

    if not result:
        raise FetchError(f"All fetch tiers failed. Last error: {error}")

    # Cache result
    if use_cache and result:
        try:
            cache_path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return result


def perform_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using DuckDuckGo.
    
    Returns:
        Formatted string with search results (Title, Link, Snippet)
    """
    try:
        from duckduckgo_search import DDGS
        
        results = DDGS().text(query, max_results=max_results)
        
        if not results:
            return "No search results found."
            
        formatted = []
        for r in results:
            formatted.append(f"### [{r['title']}]({r['href']})\n{r['body']}\n")
            
        return "\n".join(formatted)
        
    except ImportError:
        return "Error: duckduckgo-search not installed."
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"
