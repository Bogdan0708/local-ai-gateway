"""Browser tier gating and fail-closed domain allowlist."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import get_domain_config, get_settings
from src.web_fetcher import DomainValidator, SecurityError, WebFetcher

PUBLIC_A = "93.184.216.34"


@pytest.fixture(autouse=True)
def _clear_domain_cache():
    get_domain_config.cache_clear()
    yield
    get_domain_config.cache_clear()
    get_settings.cache_clear()


# === Domain allowlist ===


def test_example_domain_allowlist_is_shipped_and_loaded(settings_env):
    settings_env(ALLOW_ALL_DOMAINS="false")
    config = get_domain_config()
    assert config.allowed, "clean checkout must load an allowlist"
    assert "config/allowed_domains.example.yaml" in str(config.config_path).replace(
        "\\", "/"
    )


def test_off_list_domain_is_refused_on_clean_checkout(settings_env):
    settings_env(ALLOW_ALL_DOMAINS="false")
    with pytest.raises(SecurityError, match="whitelist|allowlist"):
        DomainValidator().validate("http://not-on-the-list.example/")


def test_listed_domain_is_accepted(settings_env):
    settings_env(ALLOW_ALL_DOMAINS="false")
    allowed = get_domain_config().allowed
    first = allowed[0].lstrip("*.")
    DomainValidator().validate(f"https://{first}/some/page")


def test_empty_allowlist_denies_by_default(settings_env, monkeypatch):
    settings_env(ALLOW_ALL_DOMAINS="false")
    validator = DomainValidator()
    monkeypatch.setattr(validator.config, "_allowed", [])
    with pytest.raises(SecurityError):
        validator.validate("https://anything.example/")


def test_empty_allowlist_allows_when_explicitly_opted_in(settings_env, monkeypatch):
    settings_env(ALLOW_ALL_DOMAINS="true")
    validator = DomainValidator()
    monkeypatch.setattr(validator.config, "_allowed", [])
    validator.validate("https://anything.example/")


def test_blocklist_still_wins_over_allow_all(settings_env, monkeypatch):
    settings_env(ALLOW_ALL_DOMAINS="true")
    validator = DomainValidator()
    monkeypatch.setattr(validator.config, "_blocked", ["evil.example"])
    with pytest.raises(SecurityError, match="blocked"):
        validator.validate("https://evil.example/")


# === Browser tier ===


def test_browser_tier_is_off_by_default(settings_env):
    settings = settings_env(ALLOW_ALL_DOMAINS="true")
    assert settings.allow_browser_tier is False


def test_browser_tier_is_not_reached_unless_enabled(settings_env):
    """max_tier=3 from a caller must not launch a browser by default."""
    settings_env(ALLOW_ALL_DOMAINS="true", ALLOW_BROWSER_TIER="false")
    fetcher = WebFetcher()
    try:
        with patch.object(fetcher, "_fetch_playwright") as playwright:
            with patch.object(
                fetcher, "_fetch_httpx", side_effect=RuntimeError("tier 1 down")
            ):
                with patch.object(
                    fetcher, "_fetch_curl", side_effect=RuntimeError("tier 2 down")
                ):
                    with pytest.raises(Exception):
                        fetcher.fetch(
                            f"http://{PUBLIC_A}/", use_cache=False, max_tier=3
                        )
        playwright.assert_not_called()
    finally:
        fetcher.close()


def test_browser_tier_runs_when_enabled(settings_env):
    settings_env(ALLOW_ALL_DOMAINS="true", ALLOW_BROWSER_TIER="true")
    fetcher = WebFetcher()
    try:
        with patch.object(fetcher, "_fetch_playwright") as playwright:
            playwright.side_effect = RuntimeError("no browser in CI")
            with patch.object(
                fetcher, "_fetch_httpx", side_effect=RuntimeError("tier 1 down")
            ):
                with patch.object(
                    fetcher, "_fetch_curl", side_effect=RuntimeError("tier 2 down")
                ):
                    with pytest.raises(Exception):
                        fetcher.fetch(
                            f"http://{PUBLIC_A}/", use_cache=False, max_tier=3
                        )
        playwright.assert_called_once()
    finally:
        fetcher.close()


def test_browser_request_guard_aborts_internal_navigation(settings_env):
    """Every browser-initiated request is validated before it is issued."""
    settings_env(ALLOW_ALL_DOMAINS="true", ALLOW_BROWSER_TIER="true")
    fetcher = WebFetcher()
    try:
        blocked_route, blocked_request = MagicMock(), MagicMock()
        blocked_request.url = "http://169.254.169.254/latest/meta-data/"
        fetcher._guard_browser_route(blocked_route, blocked_request)
        blocked_route.abort.assert_called_once()
        blocked_route.continue_.assert_not_called()

        mapped_route, mapped_request = MagicMock(), MagicMock()
        mapped_request.url = "http://[::ffff:169.254.169.254]/latest/"
        fetcher._guard_browser_route(mapped_route, mapped_request)
        mapped_route.abort.assert_called_once()

        ok_route, ok_request = MagicMock(), MagicMock()
        ok_request.url = f"http://{PUBLIC_A}/page"
        fetcher._guard_browser_route(ok_route, ok_request)
        ok_route.continue_.assert_called_once()
        ok_route.abort.assert_not_called()
    finally:
        fetcher.close()


def test_browser_tier_installs_the_guard_before_navigating(settings_env):
    """page.route(...) must be wired before page.goto(...) is called."""
    settings_env(ALLOW_ALL_DOMAINS="true", ALLOW_BROWSER_TIER="true")
    fetcher = WebFetcher()
    order = []

    page = MagicMock()
    page.url = f"http://{PUBLIC_A}/"
    page.content.return_value = "<html><body><p>ok</p></body></html>"
    page.route.side_effect = lambda *a, **k: order.append("route")
    response = MagicMock()
    response.status = 200
    page.goto.side_effect = lambda *a, **k: (order.append("goto"), response)[1]

    browser = MagicMock()
    browser.new_page.return_value = page
    playwright = MagicMock()
    playwright.chromium.launch.return_value = browser
    context = MagicMock()
    context.__enter__.return_value = playwright
    context.__exit__.return_value = False

    fake_module = MagicMock()
    fake_module.sync_playwright.return_value = context

    try:
        with patch.dict("sys.modules", {"playwright.sync_api": fake_module}):
            result = fetcher._fetch_playwright(f"http://{PUBLIC_A}/")
        assert order == ["route", "goto"]
        assert result.tier_used == "playwright"
    finally:
        fetcher.close()
