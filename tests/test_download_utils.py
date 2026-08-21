"""Unit tests for download_utils.py — SSRF guards and size-capped streaming download.

Regression coverage for the SSRF gap watch_runner._download_pdf had: it only
checked the URL scheme (http/https), never the DNS-resolved IP, so a URL whose
hostname resolves to a loopback/private/link-local address (e.g. an attacker
pointing at http://169.254.169.254/) would be fetched by the server.

Second round: the resolved-IP check and urlopen()'s own lookup were two separate
DNS queries, so a rebinding resolver could answer public for the check and
private for the fetch. Resolution now happens once and the connection is pinned
to that address; TLS still validates against the hostname.
"""

import socket
from unittest.mock import MagicMock, patch

import pytest


def _addr(ip):
    """One getaddrinfo entry for `ip`."""
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))]


# --------------------------------------------------------------------------
# Address vetting
# --------------------------------------------------------------------------
def test_resolve_rejects_loopback():
    from download_utils import _resolve_public_ip

    with patch("socket.getaddrinfo", return_value=_addr("127.0.0.1")):
        assert _resolve_public_ip("localhost", 80) is None


def test_resolve_rejects_private_range():
    from download_utils import _resolve_public_ip

    with patch("socket.getaddrinfo", return_value=_addr("10.0.0.5")):
        assert _resolve_public_ip("internal.example", 80) is None


def test_resolve_rejects_link_local():
    """Covers the metadata-service SSRF class (169.254.169.254)."""
    from download_utils import _resolve_public_ip

    with patch("socket.getaddrinfo", return_value=_addr("169.254.169.254")):
        assert _resolve_public_ip("metadata.internal", 80) is None


def test_resolve_returns_the_address_for_a_public_host():
    """It must hand back the IP, not just a verdict — that address is what the
    connection is pinned to, which is what closes the TOCTOU window."""
    from download_utils import _resolve_public_ip

    with patch("socket.getaddrinfo", return_value=_addr("93.184.216.34")):
        assert _resolve_public_ip("example.com", 443) == "93.184.216.34"


def test_resolve_dns_failure_is_not_an_error():
    """A hostname that fails to resolve isn't itself an SSRF vector — the fetch
    would simply fail. It must return None rather than raise."""
    from download_utils import _resolve_public_ip

    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        assert _resolve_public_ip("nonexistent.invalid", 80) is None


def test_a_host_resolving_to_both_public_and_private_is_refused():
    """That mix IS the rebinding pattern — one bad answer condemns the host."""
    from download_utils import _resolve_public_ip

    mixed = _addr("127.0.0.1") + _addr("93.184.216.34")
    with patch("socket.getaddrinfo", return_value=mixed):
        assert _resolve_public_ip("rebind.example", 80) is None


@pytest.mark.parametrize("ip", ["0.0.0.0", "224.0.0.1", "::1", "fe80::1", "192.168.1.1"])
def test_blocked_ip_families(ip):
    import ipaddress
    from download_utils import _is_blocked_ip

    assert _is_blocked_ip(ipaddress.ip_address(ip)) is True


def test_public_ip_is_not_blocked():
    import ipaddress
    from download_utils import _is_blocked_ip

    assert _is_blocked_ip(ipaddress.ip_address("93.184.216.34")) is False


# --------------------------------------------------------------------------
# download_pdf
# --------------------------------------------------------------------------
def test_download_pdf_rejects_non_http_scheme():
    from download_utils import download_pdf

    assert download_pdf("file:///etc/passwd") is None
    assert download_pdf("ftp://example.com/x.pdf") is None


def test_download_pdf_rejects_private_ip_target():
    import download_utils

    with patch("download_utils._resolve_public_ip", return_value=None), \
         patch("download_utils._pinned_opener") as mock_opener:
        result = download_utils.download_pdf("http://169.254.169.254/x.pdf")

    assert result is None
    mock_opener.assert_not_called()  # must reject before ever making the request


def _mock_opener(**open_kwargs):
    """A _pinned_opener stand-in whose .open() behaves as configured."""
    opener = MagicMock()
    opener.open = MagicMock(**open_kwargs)
    return opener


def test_download_pdf_streams_public_url():
    import download_utils

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"PDF-bytes", b""]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("download_utils._resolve_public_ip", return_value="93.184.216.34"), \
         patch("download_utils._pinned_opener", return_value=_mock_opener(return_value=mock_resp)):
        path = download_utils.download_pdf("http://example.com/paper.pdf")

    assert path is not None
    with open(path, "rb") as f:
        assert f.read() == b"PDF-bytes"


def test_connection_is_pinned_to_the_vetted_address():
    """The resolved IP must be what the opener is built for. If the opener were
    built without it, urlopen would resolve again and the TOCTOU hole reopens."""
    import download_utils

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"x", b""]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("download_utils._resolve_public_ip", return_value="93.184.216.34"), \
         patch("download_utils._pinned_opener",
               return_value=_mock_opener(return_value=mock_resp)) as mock_builder:
        download_utils.download_pdf("https://example.com/paper.pdf")

    mock_builder.assert_called_once_with("93.184.216.34")


def test_tls_still_validates_against_the_hostname_not_the_ip():
    """Pinning must not turn into cert-validation-by-IP, which would break every
    ordinary HTTPS fetch. The connection keeps the hostname for SNI/verification."""
    from download_utils import _PinnedHTTPSConnection

    conn = _PinnedHTTPSConnection("example.com", pinned_ip="93.184.216.34")
    assert conn.host == "example.com"
    assert conn._pinned_ip == "93.184.216.34"


def test_download_pdf_rejects_redirect_to_private_ip():
    """urlopen follows redirects by default, so a public URL that 302s to an
    internal address must not be silently followed — that fully defeats the guard."""
    import urllib.error
    import download_utils

    def fake_open(req, timeout=30):
        raise urllib.error.HTTPError(
            req.full_url, 302, "Found",
            {"Location": "http://169.254.169.254/latest/meta-data/"}, None,
        )

    def fake_resolve(hostname, port):
        # The original host looks public; the link-local redirect target is refused.
        return None if hostname == "169.254.169.254" else "93.184.216.34"

    with patch("download_utils._resolve_public_ip", side_effect=fake_resolve), \
         patch("download_utils._pinned_opener",
               return_value=_mock_opener(side_effect=fake_open)):
        result = download_utils.download_pdf("http://public-looking.example.com/redirect")

    assert result is None


def test_download_pdf_follows_redirect_to_public_url():
    """A redirect to another public URL is fine — only the private hop is blocked,
    and the new host is re-resolved and re-vetted on its own."""
    import urllib.error
    import download_utils

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.side_effect = [b"PDF-bytes", b""]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    calls = {"n": 0}

    def fake_open(req, timeout=30):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 302, "Found",
                {"Location": "http://public2.example.com/paper.pdf"}, None,
            )
        return mock_resp

    with patch("download_utils._resolve_public_ip", return_value="93.184.216.34"), \
         patch("download_utils._pinned_opener",
               return_value=_mock_opener(side_effect=fake_open)):
        path = download_utils.download_pdf("http://public1.example.com/redirect")

    assert path is not None
    assert calls["n"] == 2  # followed exactly one redirect, re-validated the new host


def test_download_pdf_size_cap_aborts_oversized_response():
    import download_utils

    big_chunk = b"x" * (download_utils._MAX_PDF_BYTES + 1)
    mock_resp = MagicMock()
    mock_resp.read.side_effect = [big_chunk, b""]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("download_utils._resolve_public_ip", return_value="93.184.216.34"), \
         patch("download_utils._pinned_opener", return_value=_mock_opener(return_value=mock_resp)):
        result = download_utils.download_pdf("http://example.com/huge.pdf")

    assert result is None
