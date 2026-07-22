"""Shared SSRF-guarded PDF downloader, used by both watches and frictionless ingestion.

Extracted from watch_runner._download_pdf, which only rejected non-http(s)
schemes — a URL whose hostname resolves to a loopback/private/link-local
address (127.0.0.1, 169.254.169.254, an internal 10.x service, ...) still got
fetched. This adds the missing DNS-resolved private-IP check.
"""

from urllib.parse import urljoin, urlparse
import ipaddress
import logging
import os
import socket
import tempfile
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB cap — guards against OOM on hostile URLs
_MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to auto-follow redirects — urlopen()'s default opener follows
    them transparently, which would let a public (SSRF-guard-passing) URL 302
    to an internal address and defeat _is_private_ip entirely. Each hop is
    re-validated by download_pdf() instead."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NoRedirectOpener = urllib.request.build_opener(_NoRedirectHandler)


def _is_private_ip(hostname: str) -> bool:
    """Reject loopback, link-local, and private IPs after DNS resolution.

    A hostname that fails to resolve is not itself an SSRF vector (the
    subsequent urlopen() call will simply fail on it), so treat resolution
    failure as "not private" rather than raising.
    """
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for family, _, _, _, sockaddr in addrs:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved:
            return True
    return False


def download_pdf(url: str, _redirects_left: int = _MAX_REDIRECTS) -> str | None:
    """Fetch a PDF to a temp file; return its path, or None on failure.

    Rejects non-HTTP(S) URLs (blocks file://, ftp://, gopher:// SSRF vectors),
    rejects URLs whose hostname resolves to a private/loopback/link-local IP,
    streams with a hard size cap so a huge response can't OOM the process, and
    manually re-validates each hop of a redirect chain (auto-following would
    let a public, guard-passing URL 302 to an internal address and bypass the
    check entirely).

    # ponytail: the private-IP check and urlopen()'s own DNS resolution are two
    # separate lookups (TOCTOU) — a malicious/rebinding DNS server could return
    # a public IP for the check and a private one for the actual fetch. Full
    # fix is connection pinning (resolve once, connect to that exact IP); add
    # it if this is ever exposed to untrusted third-party DNS at scale.
    # ponytail: timeout=30 bounds each individual connect/read, not the total
    # transfer — a slow-drip server can hold the connection open indefinitely
    # under the byte cap. Add a wall-clock ceiling across the whole read loop
    # if this becomes a real self-DoS problem (route is auth'd + rate-limited).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(f"Rejected non-HTTP(S) URL: {url}")
        return None
    if _is_private_ip(parsed.hostname or ""):
        logger.warning(f"Rejected private/loopback URL: {url}")
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "IndicRAG/2.0"})
    try:
        resp = _NoRedirectOpener.open(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in _REDIRECT_CODES:
            location = e.headers.get("Location") if e.headers else None
            if not location or _redirects_left <= 0:
                logger.warning(f"Redirect blocked or exhausted: {url}")
                return None
            return download_pdf(urljoin(url, location), _redirects_left - 1)
        logger.warning(f"PDF download failed {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"PDF download failed {url}: {e}")
        return None

    # Only create the temp file once the connection is actually open, so a
    # rejected/redirected request never leaves an unclosed fd or a stray file.
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with resp, os.fdopen(fd, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_PDF_BYTES:
                    logger.warning(f"PDF too large (>{_MAX_PDF_BYTES} bytes), aborting: {url}")
                    f.close()
                    os.unlink(path)
                    return None
                f.write(chunk)
        return path
    except Exception as e:
        logger.warning(f"PDF download failed {url}: {e}")
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
