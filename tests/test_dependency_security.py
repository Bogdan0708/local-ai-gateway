"""Security checks for the TLS trust and PDF parsing used by the gateway."""

import hashlib
import subprocess
import sys
import textwrap

import httpx
import pytest


def test_outbound_tls_does_not_trust_compromised_emsign_chain():
    # GHSA-6q48-6c7w-9gw2: compromised intermediates chain to this root.
    affected_root_sha256 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
    context = httpx.create_ssl_context(verify=True, trust_env=False)
    trusted_roots = {
        hashlib.sha256(certificate).hexdigest()
        for certificate in context.get_ca_certs(binary_form=True)
    }

    assert trusted_roots, "Outbound TLS must retain a populated trust store"
    assert affected_root_sha256 not in trusted_roots


def test_malformed_pdf_xref_is_rejected_without_hanging():
    # Reproducer from the pypdf maintainer's fix for GHSA-39mj-jf6j-2qj3:
    # https://github.com/py-pdf/pypdf/pull/4170
    body = (
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
    )
    tail = b"\nstartxref\n%d\n%%%%EOF" % len(body)
    malformed_pdf = body + b"xref\n0 3\n0000000000 65535 f \n" + tail
    parser = textwrap.dedent("""\
        import sys
        try:
            import resource
        except ImportError:
            pass
        else:
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)

        from io import BytesIO
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            PdfReader(BytesIO(sys.stdin.buffer.read()))
        except PdfReadError:
            print("rejected-malformed-xref")
        else:
            sys.exit("malformed xref was accepted")
    """)

    # Keep a vulnerable dependency from hanging the test runner itself.
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", parser],
            input=malformed_pdf,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("PDF parser hung while reading an incomplete xref table")

    assert result.returncode == 0, (
        f"PDF parser did not reject malformed input safely: exit={result.returncode}, "
        f"stderr={result.stderr.decode(errors='replace')}"
    )
    assert result.stdout.splitlines() == [b"rejected-malformed-xref"]
