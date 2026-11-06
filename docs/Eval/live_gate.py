#!/usr/bin/env python3
"""Live-pipeline eval gate on the fixture corpus (master plan P0-EVL-1).

    python docs/Eval/live_gate.py                  # gate: fail on regression vs the baseline
    python docs/Eval/live_gate.py --write-baseline # measure and commit a new baseline
    python docs/Eval/live_gate.py --selftest       # clean run passes AND planted regression fails

Steps: verify the checksum-pinned fixtures, build (or reuse) an isolated index
for them, run run_live.py retrieval-only over the suite, then evaluate.py
--baseline with the paired bootstrap gate. The index lives under
docs/Eval/.fixture_index/<key>, keyed by fixture checksum + embedding model +
CHUNKER_VERSION (§16.1), and never touches chroma_db/ or sessions.db.

The planted regression takes the query embedder down (run_live.py --plant
embedder_down), so retrieval falls back to sparse-only BM25: the selftest proves the gate can fail.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / "fixtures"
SUITE = HERE / "suites" / "fixture_v1.json"
BASELINE = HERE / "baselines" / "fixture_v1.json"
INDEX_ROOT = HERE / ".fixture_index"
TOP_K = 10

sys.path.insert(0, str(FIXTURES))
import fetch  # noqa: E402


def _index_env(index_dir: Path) -> dict:
    """Environment that points every piece of state at the fixture index."""
    env = dict(os.environ)
    env.update({
        "CHROMA_DB_DIR": str(index_dir / "chroma"),
        "BM25_CACHE_DIR": str(index_dir / "chroma"),
        "SESSIONS_DB_PATH": str(index_dir / "sessions.db"),
        "ENRICH_METADATA": "false",            # no network after fetch; titles stay deterministic
        "ENABLE_MULTIMODAL_INGEST": "false",
        "USE_HYDE": "false",
        "USE_COLBERT_RERANK": "false",
        "USE_HYBRID_SEARCH": "true",
        "PYTHONHASHSEED": "0",
    })
    return env


def _py(code_or_args: list, env: dict, check: bool = True) -> subprocess.CompletedProcess:
    print("  $ python " + " ".join(code_or_args[:3]) + (" ..." if len(code_or_args) > 3 else ""), flush=True)
    return subprocess.run([sys.executable, *code_or_args], env=env, cwd=ROOT, check=check)  # NOSONAR S8705: argv list, no shell; repo scripts + operator paths


def index_key() -> str:
    """fixture checksum + embed model + chunker version, read from the code itself."""
    out = subprocess.run(
        [sys.executable, "-c", "import config, vector_store; "
         "print(config.EMBEDDING_MODEL_NAME.replace('/', '_'), vector_store.CHUNKER_VERSION)"],
        cwd=ROOT, capture_output=True, text=True, check=True, env=_index_env(INDEX_ROOT / "_probe"))
    model, chunker = out.stdout.split()[-2:]
    return f"{fetch.fixture_checksum()}_{model}_c{chunker}"


_INGEST = """
import json, sys
from pathlib import Path
import ingest, vector_store
stats = ingest.ingest_directory(sys.argv[1])
col = vector_store.get_or_create_collection()
got = vector_store.get_chunks(col, include=["metadatas"], timeout=None)
ids = sorted({m.get("paper_id") for m in got["metadatas"]})
print(json.dumps({"stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
                  "chunks": vector_store.count(col, timeout=None), "paper_ids": ids}))
"""


def ensure_index(index_dir: Path) -> dict:
    ready = index_dir / "READY.json"
    if ready.exists():
        info = json.loads(ready.read_text("utf-8"))
        print(f"index: reusing {index_dir.name} ({info['chunks']} chunks)")
        return info
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"index: building {index_dir.name} (cold ingest of the fixture corpus)", flush=True)
    res = subprocess.run([sys.executable, "-c", _INGEST, str(FIXTURES / "pdfs")],
                         env=_index_env(index_dir), cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-4000:])
        raise SystemExit("index: ingest failed")
    info = json.loads(res.stdout.strip().splitlines()[-1])
    expected = sorted(p["id"] for p in fetch.load_manifest())
    if info["paper_ids"] != expected or info["chunks"] == 0:
        # A partial index would score as a regression; refuse it instead.
        missing = sorted(set(expected) - set(info["paper_ids"]))
        raise SystemExit(f"index: incomplete ({info['chunks']} chunks; missing {missing})")
    ready.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"index: ready ({info['chunks']} chunks from {len(expected)} papers)")
    return info


def run_suite(index_dir: Path, out: Path, planted: bool) -> int:
    """Run the suite; return how many results came back degraded."""
    args = [str(HERE / "run_live.py"), "--judgments", str(SUITE), "--out", str(out), "--top-k", str(TOP_K)]
    if planted:
        args += ["--plant", "embedder_down"]
    _py(args, _index_env(index_dir))
    results = json.loads(out.read_text("utf-8"))["results"]  # NOSONAR S8707: operator-supplied CLI path in a developer eval tool
    return sum(1 for r in results if r.get("degraded"))


def score(results: Path, metrics_out: Path, baseline: Path = None) -> int:
    args = [str(HERE / "evaluate.py"), "--judgments", str(SUITE), "--results", str(results),
            "--k", "5", "--json", str(metrics_out)]
    if baseline:
        args += ["--baseline", str(baseline)]
    return _py(args, dict(os.environ), check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write-baseline", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    ap.add_argument("--planted", action="store_true", help="run the planted dense-only regression")
    ap.add_argument("--work", default=str(INDEX_ROOT / "runs"), help="where run outputs go")
    args = ap.parse_args()

    problems = fetch.verify(fetch.load_manifest(), FIXTURES / "pdfs")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit("fixtures missing or modified: run docs/Eval/fixtures/fetch.py")
    index_dir = INDEX_ROOT / index_key()
    ensure_index(index_dir)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)  # NOSONAR S8707: operator-supplied CLI path in a developer eval tool

    if args.write_baseline:
        if run_suite(index_dir, work / "clean.json", planted=False):
            raise SystemExit("refusing to write a baseline from degraded retrieval")
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        rc = score(work / "clean.json", BASELINE)
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return rc

    if not BASELINE.exists():
        raise SystemExit(f"no baseline at {BASELINE}: run with --write-baseline first")

    if args.selftest:
        clean_degraded = run_suite(index_dir, work / "clean.json", planted=False)
        clean_rc = score(work / "clean.json", work / "clean_metrics.json", BASELINE)
        planted_degraded = run_suite(index_dir, work / "planted.json", planted=True)
        if not planted_degraded:
            raise SystemExit("selftest: the planted fault did not take effect")
        # The METRIC gate must catch it on its own; the degraded check is a second line.
        planted_rc = score(work / "planted.json", work / "planted_metrics.json", BASELINE)
        clean = "pass" if clean_rc == 0 and not clean_degraded else "fail"
        planted = "fail" if planted_rc == 1 else "pass"
        print(f"\nselftest: clean={clean} planted={planted}")
        if clean == "pass" and planted == "fail":
            print("LIVE_GATE_SELFTEST_OK clean=pass planted=fail")
            return 0
        return 1

    degraded = run_suite(index_dir, work / "run.json", planted=args.planted)
    rc = score(work / "run.json", work / "metrics.json", BASELINE)
    if degraded:
        # Metrics alone could pass a broken pipeline on an easy query; a degraded leg never passes CI.
        print(f"\n  CI FAIL: {degraded} result(s) came back degraded (sparse-only)")
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
