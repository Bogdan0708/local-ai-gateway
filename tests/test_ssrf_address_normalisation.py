"""IPv6 forms that embed an IPv4 address must be validated on the IPv4 too.

`::ffff:169.254.169.254`, NAT64 `64:ff9b::7f00:1` and 6to4 `2002:7f00:1::1`
all reach 169.254.169.254 / 127.0.0.1, but none of them is a member of the
IPv4 networks in BLOCKED_IP_NETWORKS.
"""

import ipaddress
import socket
from unittest.mock import patch

import httpx
import pytest

from src.web_fetcher import SafeDNSResolver, SecurityError, WebFetcher

PUBLIC_A = "93.184.216.34"

# (literal, what it really reaches)
EMBEDDED_INTERNAL = [
    ("::ffff:127.0.0.1", "IPv4-mapped loopback"),
    ("::ffff:169.254.169.254", "IPv4-mapped cloud metadata"),
    ("::ffff:10.0.0.5", "IPv4-mapped RFC1918"),
    ("64:ff9b::7f00:1", "NAT64 loopback"),
    ("64:ff9b::a9fe:a9fe", "NAT64 cloud metadata"),
    ("2002:7f00:1::1", "6to4 loopback"),
    ("2002:a9fe:a9fe::1", "6to4 cloud metadata"),
    ("64:ff9b:1::7f00:1", "NAT64 local-use prefix"),
]


@pytest.fixture
def fetcher(settings_env):
    settings_env(ALLOW_ALL_DOMAINS="true")
    f = WebFetcher()
    f.ssrf_validator.clear_dns_cache()
    yield f
    f.close()


@pytest.mark.parametrize("literal,description", EMBEDDED_INTERNAL)
def test_validate_ip_rejects_embedded_internal_addresses(literal, description):
    resolver = SafeDNSResolver()
    with pytest.raises(SecurityError):
        resolver._validate_ip(literal, "synthetic.example.test")


def test_validate_ip_still_accepts_public_addresses():
    resolver = SafeDNSResolver()
    resolver._validate_ip(PUBLIC_A, "example.test")
    resolver._validate_ip("2606:2800:220:1:248:1893:25c8:1946", "example.test")
    resolver._validate_ip(f"::ffff:{PUBLIC_A}", "example.test")


@pytest.mark.parametrize("literal,description", EMBEDDED_INTERNAL)
def test_initial_url_with_embedded_internal_address_is_blocked(
    fetcher, literal, description
):
    """No request may be issued at all for such a URL."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="<html><body><p>leaked</p></body></html>")

    fetcher._http_client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(SecurityError):
        fetcher.fetch(f"http://[{literal}]/", use_cache=False, max_tier=1)

    assert seen == []


@pytest.mark.parametrize("literal,description", EMBEDDED_INTERNAL)
def test_redirect_to_embedded_internal_address_is_blocked(
    fetcher, literal, description
):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(
                302, headers={"location": f"http://[{literal}]/synthetic-internal"}
            )
        return httpx.Response(
            200,
            text="<html><body><p>Synthetic internal result</p></body></html>",
            headers={"content-type": "text/html"},
        )

    fetcher._http_client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=True
    )

    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_A, 80))],
    ):
        with pytest.raises(SecurityError):
            fetcher.fetch("http://example.test/", use_cache=False, max_tier=1)

    # Only the first, public hop was ever requested.
    assert len(seen) == 1
    assert literal not in seen[0]


def test_dns_answer_with_embedded_internal_address_is_blocked(fetcher):
    """A hostile DNS answer cannot launder loopback through an IPv6 form."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:127.0.0.1", 80, 0, 0))
        ]

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        with pytest.raises(SecurityError):
            fetcher.fetch("http://rebind.example.test/", use_cache=False, max_tier=1)


def test_ipv6_request_url_is_bracketed():
    """Sanity check on the URL rebuild used for validated IPv6 destinations."""
    resolver = SafeDNSResolver()
    url, host = resolver.build_url_with_ip(
        "http://example.test/x", "2606:2800:220:1:248:1893:25c8:1946"
    )
    assert url.startswith("http://[2606:2800:220:1:248:1893:25c8:1946]/")
    assert host == "example.test"


def test_normalisation_helper_covers_known_forms():
    from src.web_fetcher import embedded_addresses

    def targets(literal):
        return {
            str(addr)
            for addr, _ in embedded_addresses(ipaddress.ip_address(literal))
        }

    assert "127.0.0.1" in targets("::ffff:127.0.0.1")
    assert "127.0.0.1" in targets("64:ff9b::7f00:1")
    assert "127.0.0.1" in targets("2002:7f00:1::1")
