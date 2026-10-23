"""Real HTTPX/curl TLS handshakes against a private, local test server."""

from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shutil
import socket
import ssl
import threading
from types import SimpleNamespace

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import httpx
import pytest

from src.web_fetcher import FetchError, WebFetcher, _Redirect


@pytest.fixture(params=["127.0.0.1", "::1"])
def tls_server(request, tmp_path, monkeypatch):
    address = request.param
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Gateway test CA")])
    now = datetime.now(timezone.utc)

    def certificate(subject, public_key):
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
        )

    ca = (
        certificate(ca_name, ca_key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    server_cert = (
        certificate(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gateway.test")]),
            server_key.public_key(),
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("gateway.test"), x509.DNSName("xn--fa-hia.test"),
        ]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.KeyUsage(True, False, True, False, False, False, False, False, False),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_file, cert_file, key_file = (tmp_path / name for name in ("ca.pem", "server.pem", "key.pem"))
    ca_file.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_file.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(server_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(ca_file))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    requests, server_names = [], []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (ConnectionError, ssl.SSLError):
                pass  # A client rejecting our test certificate closes the socket.

        def do_GET(self):
            requests.append({"path": self.path, "host": self.headers.get("Host"),
                             "authorization": self.headers.get("Authorization")})
            body = b"<html><body><p>Pinned TLS works</p></body></html>"
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "http://127.0.0.1/private")
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in address else socket.AF_INET
        daemon_threads = True

    try:
        server = Server((address, 0), Handler)
    except OSError:
        if ":" in address:
            pytest.skip("IPv6 loopback is unavailable")
        raise
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    context.set_servername_callback(lambda sock, name, ctx: server_names.append(name))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    ip_authority = f"[{address}]" if ":" in address else address
    try:
        yield SimpleNamespace(
            pinned_url=f"https://{ip_authority}:{port}/",
            original_url=f"https://gateway.test:{port}/",
            port=port, requests=requests, server_names=server_names,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture(params=["httpx", "curl"])
def fetch(request, tls_server, settings_env):
    if request.param == "curl" and not shutil.which("curl"):
        pytest.skip("curl is unavailable")
    settings_env(WEB_FETCH_TIMEOUT=2)

    def run(path="", hostname="gateway.test", credentials=""):
        original = f"https://{credentials}{hostname}:{tls_server.port}/{path}"
        fetcher = WebFetcher()
        try:
            method = getattr(fetcher, f"_fetch_{request.param}")
            return method(tls_server.pinned_url + path, hostname, original)
        finally:
            fetcher.close()

    return run


def test_pinned_https_uses_original_tls_name_and_http_authority(fetch, tls_server):
    result = fetch()
    assert result.status_code == 200
    assert "Pinned TLS works" in result.content
    assert tls_server.server_names == ["gateway.test"]
    assert tls_server.requests == [{"path": "/", "host": f"gateway.test:{tls_server.port}",
                                    "authorization": None}]


def test_pinned_https_rejects_mismatched_hostname(fetch, tls_server):
    with pytest.raises((httpx.HTTPError, FetchError), match="(?i)certificate|SSL"):
        fetch(hostname="wrong.test")
    assert tls_server.requests == []


def test_pinned_https_uses_one_idna_name_for_tls_and_http(fetch, tls_server):
    result = fetch(hostname="faß.test")
    assert result.status_code == 200
    assert tls_server.server_names == ["xn--fa-hia.test"]
    assert tls_server.requests[0]["host"] == f"xn--fa-hia.test:{tls_server.port}"


def test_pinned_https_rejects_untrusted_certificate(fetch, tls_server, monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.delenv("CURL_CA_BUNDLE")
    with pytest.raises((httpx.HTTPError, FetchError), match="(?i)certificate|SSL"):
        fetch()
    assert tls_server.requests == []


def test_pinned_https_does_not_follow_redirects(fetch, tls_server):
    result = fetch(path="redirect")
    assert isinstance(result, _Redirect)
    assert result.location == "http://127.0.0.1/private"
    assert [request["path"] for request in tls_server.requests] == ["/redirect"]


def test_pinned_https_ignores_proxy_environment(fetch, tls_server, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
        monkeypatch.setenv(name.lower(), "http://127.0.0.1:1")
    result = fetch()
    assert result.status_code == 200
    assert [request["path"] for request in tls_server.requests] == ["/"]


def test_pinned_https_does_not_forward_url_credentials(fetch, tls_server):
    result = fetch(credentials="user:synthetic-password@")
    assert result.status_code == 200
    assert tls_server.requests[0]["authorization"] is None


def test_httpx_rechecks_tls_name_when_two_hosts_share_an_ip(tls_server, settings_env):
    settings_env(WEB_FETCH_TIMEOUT=2)
    fetcher = WebFetcher()
    try:
        result = fetcher._fetch_httpx(tls_server.pinned_url, "gateway.test", tls_server.original_url)
        assert result.status_code == 200
        with pytest.raises(httpx.ConnectError, match="(?i)certificate|SSL"):
            fetcher._fetch_httpx(
                tls_server.pinned_url, "wrong.test", f"https://wrong.test:{tls_server.port}/",
            )
        assert [request["host"] for request in tls_server.requests] == [f"gateway.test:{tls_server.port}"]
    finally:
        fetcher.close()


def test_curl_ignores_config_that_disables_tls_verification(tls_server, settings_env, tmp_path, monkeypatch):
    if not shutil.which("curl"):
        pytest.skip("curl is unavailable")
    (tmp_path / ".curlrc").write_text("insecure\n")
    monkeypatch.setenv("CURL_HOME", str(tmp_path))
    settings_env(WEB_FETCH_TIMEOUT=2)
    fetcher = WebFetcher()
    try:
        with pytest.raises(FetchError, match="(?i)certificate|SSL"):
            fetcher._fetch_curl(
                tls_server.pinned_url, "wrong.test", f"https://wrong.test:{tls_server.port}/",
            )
        assert tls_server.requests == []
    finally:
        fetcher.close()
