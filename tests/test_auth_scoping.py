"""Regression tests for the two authorization fixes from the PR #12 review.

1. `verify_admin_key` used to fall back to `verify_api_key`, so any ordinary
   user key authorized /purge/*, and nothing at all was required when API_KEYS
   was unset. It is now fail-closed on ADMIN_API_KEY.
2. Compare jobs live in the global `_jobs` store, so `GET /compare/status/{id}`
   handed any authenticated caller another user's results. Jobs now carry an
   owner fingerprint.
"""

import asyncio

import pytest
from fastapi import HTTPException

import deps


def _run(coro):
    return asyncio.run(coro)


def test_admin_key_required_when_unset(monkeypatch):
    """No ADMIN_API_KEY => destructive routes are refused, not opened up."""
    monkeypatch.setattr(deps, "_admin_key", None)
    monkeypatch.setattr(deps, "VALID_API_KEYS", {"user-key"})
    with pytest.raises(HTTPException) as exc:
        _run(deps.verify_admin_key("user-key"))
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "ADMIN_KEY_NOT_CONFIGURED"


def test_admin_key_refused_when_unauthenticated(monkeypatch):
    """ALLOW_UNAUTHENTICATED (no API_KEYS) must not make /purge/* anonymous."""
    monkeypatch.setattr(deps, "_admin_key", None)
    monkeypatch.setattr(deps, "VALID_API_KEYS", None)
    with pytest.raises(HTTPException) as exc:
        _run(deps.verify_admin_key(None))
    assert exc.value.status_code == 403


def test_user_key_does_not_authorize_admin(monkeypatch):
    monkeypatch.setattr(deps, "_admin_key", "admin-key")
    monkeypatch.setattr(deps, "VALID_API_KEYS", {"user-key", "admin-key"})
    with pytest.raises(HTTPException) as exc:
        _run(deps.verify_admin_key("user-key"))
    assert exc.value.detail["code"] == "ADMIN_KEY_REQUIRED"
    assert _run(deps.verify_admin_key("admin-key")) is True


def test_owner_fingerprint_distinguishes_keys(monkeypatch):
    monkeypatch.setattr(deps, "VALID_API_KEYS", {"key-a", "key-b"})
    a = _run(deps.current_owner("key-a"))
    b = _run(deps.current_owner("key-b"))
    assert a and b and a != b
    assert a == _run(deps.current_owner("key-a")), "fingerprint must be stable"
    assert "key-a" not in a, "fingerprint must not embed the key itself"


def test_owner_fingerprint_is_none_without_auth(monkeypatch):
    """Single-tenant mode: nothing to scope against, so no owner is recorded."""
    monkeypatch.setattr(deps, "VALID_API_KEYS", None)
    assert _run(deps.current_owner("anything")) is None
