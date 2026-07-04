"""Test isolation: point SQLite session/job persistence at a throwaway file.

Must run before any test module imports config/persistence/deps/api_server,
so this sets the env var at conftest module level (pytest imports conftest.py
files before collecting test modules) — a fixture would run too late.
"""

import os
import tempfile

_tmp_db = os.path.join(tempfile.gettempdir(), "indicrag_test_sessions.db")
os.environ["SESSIONS_DB_PATH"] = _tmp_db

for suffix in ("", "-wal", "-shm"):
    try:
        os.remove(_tmp_db + suffix)
    except FileNotFoundError:
        pass
