"""Redirect handling in the web fetcher.

Validating only the first URL is not SSRF protection: every hop must be
re-validated (scheme, hostname, freshly resolved IP) and the chain must be
bounded. No test in this file touches the network - httpx.MockTransport and a
patched socket.getaddrinfo stand in for it.
"""

import socket
from unittest.mock import patch

import httpx
import pytest

from src.web_fetcher import MAX_REDIRECTS, SecurityError, WebFetcher

PUBLIC_A = "93.184.216.34"
PUBLIC_B = "93.184.216.35"


def _redirect_chain(fetcher, hops):
    """Wire a mock transport that replies with `hops` locations, then 200 OK."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        index = len(seen) - 1
        if index < len(hops):
            return httpx.Response(302, headers={"location": hops[index]})
        return httpx.Response(
            200,
            text="<html><body><p>Synthetic final page</p></body></html>",
            headers={"content-type": "text/html"},
        )

    # follow_redirects=True here on purpose: the fetcher must refuse to follow
    # redirects even when handed a permissive client.
    fetcher._http_client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=True
    )
    return seen


def _fail_httpx(fetcher):
    """Make tier 1 fail so the curl fallback is exercised, without networking."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("synthetic tier-1 failure", request=request)

    fetcher._http_client = httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def fetcher(settings_env):
    # These tests exercise the IP-level checks; opt out of the domain
    # allowlist so synthetic literal addresses are not refused earlier.
    settings_env(ALLOW_ALL_DOMAINS="true")
    f = WebFetcher()
    f.ssrf_validator.clear_dns_cache()
    yield f
    f.close()


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/synthetic-internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/synthetic-internal",
        "http://192.168.1.10/synthetic-internal",
        "http://[::1]/synthetic-internal",
    ],
)
def test_redirect_to_internal_address_is_blocked(fetcher, target):
    seen = _redirect_chain(fetcher, [target])
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_A, 80))],
    ):
        with pytest.raises(SecurityError):
            fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=1)

    # The first hop was requested; the internal target never was.
    assert len(seen) == 1
    assert "127.0.0.1" not in seen[0]


def test_redirect_to_rebound_hostname_is_blocked(fetcher):
    """A hop to a hostname that resolves into RFC1918 space must be refused."""
    seen = _redirect_chain(fetcher, ["http://internal.example.test/admin"])

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = "10.0.0.5" if host == "internal.example.test" else PUBLIC_A
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]

    with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
        with pytest.raises(SecurityError):
            fetcher.fetch("http://public.example.test/", use_cache=False, max_tier=1)

    assert len(seen) == 1


def test_public_to_public_redirect_is_followed(fetcher):
    seen = _redirect_chain(fetcher, [f"http://{PUBLIC_B}/final"])
    result = fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=1)

    assert result.status_code == 200
    assert "Synthetic final page" in result.content
    assert len(seen) == 2
    assert result.url.endswith("/final")


def test_redirect_limit_is_enforced(fetcher):
    hops = [f"http://{PUBLIC_A}/hop{i}" for i in range(MAX_REDIRECTS + 1)]
    seen = _redirect_chain(fetcher, hops)

    with pytest.raises(SecurityError, match="redirect"):
        fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=1)

    assert len(seen) == MAX_REDIRECTS + 1


def test_redirect_at_the_limit_is_allowed(fetcher):
    hops = [f"http://{PUBLIC_A}/hop{i}" for i in range(MAX_REDIRECTS)]
    _redirect_chain(fetcher, hops)

    result = fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=1)
    assert result.status_code == 200


def test_literal_internal_ip_is_blocked_without_dns(fetcher):
    """An IP literal must be validated directly, never via getaddrinfo."""
    with patch(
        "socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_A, 80))],
    ):
        with pytest.raises(SecurityError):
            fetcher.fetch("http://127.0.0.1/synthetic", use_cache=False, max_tier=1)


def test_curl_tier_does_not_follow_redirects(fetcher):
    """The curl fallback must be invoked without -L."""
    import src.web_fetcher as wf

    _fail_httpx(fetcher)
    captured = {}

    class _Result:
        returncode = 0
        stdout = "<html><body>curl body</body></html>"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        header_file = cmd[cmd.index("-D") + 1]
        with open(header_file, "w") as f:
            f.write("HTTP/1.1 200 OK\r\ncontent-type: text/html\r\n\r\n")
        return _Result()

    with patch.object(wf.subprocess, "run", side_effect=fake_run):
        fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=2)

    cmd = captured["cmd"]
    assert "-L" not in cmd
    assert "--location" not in cmd
    assert "--max-redirs" in cmd
    assert cmd[cmd.index("--max-redirs") + 1] == "0"


def test_curl_tier_revalidates_redirect_targets(fetcher):
    """A 302 seen by curl goes through the same validation as httpx."""
    import src.web_fetcher as wf

    _fail_httpx(fetcher)

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        header_file = cmd[cmd.index("-D") + 1]
        with open(header_file, "w") as f:
            f.write(
                "HTTP/1.1 302 Found\r\nlocation: http://169.254.169.254/latest/\r\n\r\n"
            )
        return _Result()

    with patch.object(wf.subprocess, "run", side_effect=fake_run):
        with pytest.raises(SecurityError):
            fetcher.fetch(f"http://{PUBLIC_A}/", use_cache=False, max_tier=2)
