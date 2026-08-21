"""Shared SSRF-guarded PDF downloader, used by both watches and frictionless ingestion.

Extracted from watch_runner._download_pdf, which only rejected non-http(s)
schemes — a URL whose hostname resolves to a loopback/private/link-local
address (127.0.0.1, 169.254.169.254, an internal 10.x service, ...) still got
fetched. This adds the missing DNS-resolved private-IP check.
"""

from typing import Optional
from urllib.parse import urljoin, urlparse
import http.client
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


def _is_blocked_ip(ip) -> bool:
    """True for any address a fetch must never reach."""
    return bool(ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified)


def _resolve_public_ip(hostname: str, port: int) -> Optional[str]:
    """Resolve `hostname` once and return a single vetted IP to connect to.

    Returning the address — rather than a yes/no verdict — is what closes the
    TOCTOU hole. The old code asked "is this hostname private?" and then let
    urlopen() resolve the name a SECOND time, so a hostile or rebinding DNS
    server could answer public for the check and 127.0.0.1 or 169.254.169.254
    for the actual fetch. The connection is now pinned to exactly this address.

    Returns None when the name doesn't resolve (not itself an SSRF vector — the
    fetch would simply fail) or when it resolves to anything blocked.
    """
    try:
        addrs = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return None
    for _family, _type, _proto, _canon, sockaddr in addrs:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            # One bad answer condemns the host: a name resolving to both a public
            # and a private address is the rebinding pattern itself.
            return None
        return sockaddr[0]
    return None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pre-resolved IP, with TLS still verified against
    the original hostname.

    `self.host` stays the hostname so SNI, certificate validation and the Host
    header all remain correct; only the socket's destination is overridden.
    Validating the certificate against the IP instead would break every ordinary
    HTTPS fetch.
    """

    def __init__(self, host, *args, pinned_ip=None, **kwargs):
        super().__init__(host, *args, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Plain-HTTP counterpart of _PinnedHTTPSConnection."""

    def __init__(self, host, *args, pinned_ip=None, **kwargs):
        super().__init__(host, *args, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


def _pinned_opener(pinned_ip: str) -> urllib.request.OpenerDirector:
    """Opener that refuses redirects and connects only to `pinned_ip`."""
    class _HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(
                lambda host, **kw: _PinnedHTTPConnection(host, pinned_ip=pinned_ip, **kw), req)

    class _HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            # Forward this handler's SSL context rather than letting the connection
            # fall back to a default one — the handler's context is what carries
            # verification settings, and silently swapping it is how a "pinning"
            # change turns into an unverified-TLS change.
            return self.do_open(
                lambda host, **kw: _PinnedHTTPSConnection(host, pinned_ip=pinned_ip, **kw),
                req, context=self._context)

    return urllib.request.build_opener(_NoRedirectHandler, _HTTPHandler, _HTTPSHandler)


def download_pdf(url: str, _redirects_left: int = _MAX_REDIRECTS) -> str | None:
    """Fetch a PDF to a temp file; return its path, or None on failure.

    Rejects non-HTTP(S) URLs (blocks file://, ftp://, gopher:// SSRF vectors),
    rejects URLs whose hostname resolves to a private/loopback/link-local IP,
    streams with a hard size cap so a huge response can't OOM the process, and
    manually re-validates each hop of a redirect chain (auto-following would
    let a public, guard-passing URL 302 to an internal address and bypass the
    check entirely).

    DNS is resolved exactly once and the connection is pinned to that address,
    so a rebinding DNS server cannot answer public for the check and private for
    the fetch. TLS still validates against the hostname, not the pinned IP.

    # ponytail: timeout=30 bounds each individual connect/read, not the total
    # transfer — a slow-drip server can hold the connection open indefinitely
    # under the byte cap. Add a wall-clock ceiling across the whole read loop
    # if this becomes a real self-DoS problem (route is auth'd + rate-limited).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning(f"Rejected non-HTTP(S) URL: {url}")
        return None
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    pinned_ip = _resolve_public_ip(hostname, port)
    if pinned_ip is None:
        logger.warning(f"Rejected unresolvable or private/loopback URL: {url}")
        return None
    req = urllib.request.Request(url, headers={"User-Agent": "IndicRAG/2.0"})
    try:
        resp = _pinned_opener(pinned_ip).open(req, timeout=30)
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
