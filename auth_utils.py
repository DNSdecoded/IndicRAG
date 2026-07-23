"""Password hashing for multi-user login — stdlib only (pbkdf2), no new deps.

Format stored per user: a random salt + the pbkdf2-HMAC-SHA256 derivation of the
password. Verification is constant-time. Iteration count lives in config so it
can be raised later without a code change.
"""

import hashlib
import hmac
import secrets

import config


def hash_password(password: str, salt: str = None, iterations: int = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex). Generates a fresh salt unless one is given."""
    if iterations is None:
        iterations = config.PBKDF2_ITERATIONS
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), iterations)
    return salt, dk.hex()


def verify_password(password: str, salt: str, expected_hash: str,
                    iterations: int = None) -> bool:
    """Constant-time check of a password against a stored (salt, hash)."""
    if not salt or not expected_hash:
        return False
    _, actual = hash_password(password, salt, iterations)
    return hmac.compare_digest(actual, expected_hash)


def new_api_key() -> str:
    """A fresh per-user bearer key for the X-API-Key header."""
    return secrets.token_urlsafe(24)


if __name__ == "__main__":
    s, h = hash_password("hunter2")
    assert verify_password("hunter2", s, h)
    assert not verify_password("wrong", s, h)
    assert not verify_password("hunter2", s, h[:-1] + ("0" if h[-1] != "0" else "1"))
    assert len(new_api_key()) >= 24 and new_api_key() != new_api_key()
    print("auth_utils self-check ok")
