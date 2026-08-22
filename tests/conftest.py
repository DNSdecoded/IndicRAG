"""Test isolation: point on-disk state at throwaway locations.

Must run before any test module imports config/persistence/deps/api_server, so
this sets the env vars at conftest module level (pytest imports conftest.py
files before collecting test modules) — a fixture would run too late.
"""

import glob
import os
import tempfile

_tmp_db = os.path.join(tempfile.gettempdir(), "indicrag_test_sessions.db")
os.environ["SESSIONS_DB_PATH"] = _tmp_db

for suffix in ("", "-wal", "-shm"):
    try:
        os.remove(_tmp_db + suffix)
    except FileNotFoundError:
        pass

# The BM25 index cache defaults to chroma_db/, i.e. real data. Tests build
# indexes from fake collections, so without this they write fixture-derived
# files next to the live vector store — including one named after the real
# collection, which a later server start would happily load.
_tmp_bm25 = os.path.join(tempfile.gettempdir(), "indicrag_test_bm25")
os.makedirs(_tmp_bm25, exist_ok=True)
os.environ["BM25_CACHE_DIR"] = _tmp_bm25

for _stale in glob.glob(os.path.join(_tmp_bm25, "bm25_*")):
    try:
        os.remove(_stale)
    except OSError:
        pass
