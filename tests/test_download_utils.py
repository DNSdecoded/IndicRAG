"""Unit tests for download_utils.py — SSRF guards and size-capped streaming download.

Regression coverage for the SSRF gap watch_runner._download_pdf had: it only
checked the URL scheme (http/https), never the DNS-resolved IP, so a URL whose
hostname resolves to a loopback/private/link-local address (e.g. an attacker
pointing at http://169.254.169.254/ or a rebound DNS name) would be fetched
by the server. download_utils adds that check.
"""

from unittest.mock import MagicMock, patch


def test_is_private_ip_rejects_loopback():
    from download_utils import _is_private_ip

    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 0))]):
        assert _is_private_ip("localhost") is True


def test_is_private_ip_rejects_private_range():
    from download_utils import _is_private_ip

    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.5", 0))]):
        assert _is_private_ip("internal.example") is True


def test_is_private_ip_rejects_link_local():
    """Covers the metadata-service SSRF class (169.254.169.254)."""
    from download_utils import _is_private_ip

    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("169.254.169.254", 0))]):
        assert _is_private_ip("metadata.internal") is True


def test_is_private_ip_allows_public():
    from download_utils import _is_private_ip

    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
        assert _is_private_ip("example.com") is False


def test_is_private_ip_dns_failure_treated_as_safe_to_skip():
    """A hostname that fails to resolve isn't itself an SSRF vector — download_pdf's
    own urlopen() will fail on it separately; _is_private_ip shouldn't crash."""
    import socket
    from download_utils import _is_private_ip

    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        assert _is_private_ip("nonexistent.invalid") is False


def test_download_pdf_rejects_non_http_scheme():
    from download_utils import download_pdf

    assert download_pdf("file:///etc/passwd") is None
    assert download_pdf("ftp://example.com/x.pdf") is None


def test_download_pdf_rejects_private_ip_target():
    from download_utils import download_pdf
    import download_utils

    with patch("download_utils._is_private_ip", return_value=True), \
         patch.object(download_utils._NoRedirectOpener, "open") as mock_open:
        result = download_pdf("http://169.254.169.254/x.pdf")

    assert result is None
    mock_open.assert_not_called()  # must reject before ever making the request


def test_download_pdf_streams_public_url(tmp_path):
    from download_utils import download_pdf
    import download_utils

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"PDF-bytes", b""]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("download_utils._is_private_ip", return_value=False), \
         patch.object(download_utils._NoRedirectOpener, "open", return_value=mock_resp):
        path = download_pdf("http://example.com/paper.pdf")

    assert path is not None
    with open(path, "rb") as f:
        assert f.read() == b"PDF-bytes"


def test_download_pdf_rejects_redirect_to_private_ip():
    """Regression: urlopen follows redirects by default, so a public URL that
    302s to an internal address (cloud metadata service, localhost, ...) must
    not be silently followed — that fully defeats the private-IP guard."""
    import urllib.error
    import download_utils

    def fake_open(req, timeout=30):
        raise urllib.error.HTTPError(
            req.full_url, 302, "Found", {"Location": "http://169.254.169.254/latest/meta-data/"}, None
        )

    # Original host looks public and passes; the redirect target is a real
    # link-local literal, so the real (unmocked) check correctly rejects it —
    # only the original hostname's lookup needs faking here.
    def fake_is_private(hostname):
        return hostname == "169.254.169.254"

    with patch("download_utils._is_private_ip", side_effect=fake_is_private), \
         patch.object(download_utils._NoRedirectOpener, "open", side_effect=fake_open):
        result = download_utils.download_pdf("http://public-looking.example.com/redirect")

    assert result is None


def test_download_pdf_follows_redirect_to_public_url(tmp_path):
    """A redirect to another public URL is fine — only the private-IP hop is blocked."""
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
                req.full_url, 302, "Found", {"Location": "http://public2.example.com/paper.pdf"}, None
            )
        return mock_resp

    with patch("download_utils._is_private_ip", return_value=False), \
         patch.object(download_utils._NoRedirectOpener, "open", side_effect=fake_open):
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

    with patch("download_utils._is_private_ip", return_value=False), \
         patch.object(download_utils._NoRedirectOpener, "open", return_value=mock_resp):
        result = download_utils.download_pdf("http://example.com/huge.pdf")

    assert result is None
