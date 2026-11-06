#!/usr/bin/env python3
"""Fetch and verify the checksum-pinned eval fixture corpus.

    python docs/Eval/fixtures/fetch.py            # download missing PDFs, then verify
    python docs/Eval/fixtures/fetch.py --verify   # verify only (no network)

A file whose sha256 differs from manifest.json is an error, never re-used: the
gate must score exactly the corpus its baseline was measured on.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
DEFAULT_DEST = HERE / "pdfs"
_UA = {"User-Agent": "indicrag-eval-fixtures/1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["papers"]


def fixture_checksum(path: Path = MANIFEST) -> str:
    """One digest for the whole corpus: the index-cache key component (§16.1)."""
    return hashlib.sha256("".join(p["sha256"] for p in load_manifest(path)).encode()).hexdigest()[:16]


def download(paper: dict, dest: Path, tries: int = 3) -> None:
    target = dest / paper["file"]
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(paper["url"], headers=_UA)
            data = urllib.request.urlopen(req, timeout=120).read()
            break
        except Exception as e:
            if attempt == tries:
                raise RuntimeError(f"{paper['id']}: download failed: {e}") from e
            time.sleep(5 * attempt)
    if hashlib.sha256(data).hexdigest() != paper["sha256"]:
        raise RuntimeError(f"{paper['id']}: downloaded bytes do not match the pinned sha256")
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    time.sleep(3)   # arXiv asks automated clients for ~3 s between requests


def verify(papers: list, dest: Path) -> list:
    """Problems found; empty means every pinned file is present and matches."""
    problems = []
    for p in papers:
        f = dest / p["file"]
        if not f.exists():
            problems.append(f"{p['id']}: missing ({f})")
        elif sha256(f) != p["sha256"]:
            problems.append(f"{p['id']}: sha256 mismatch")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="verify only; never download")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    papers = load_manifest()
    if not args.verify:
        for p in papers:
            if not (dest / p["file"]).exists():
                print(f"fetching {p['id']}", flush=True)
                download(p, dest)
    problems = verify(papers, dest)
    for msg in problems:
        print(f"ERROR: {msg}", file=sys.stderr)
    print(f"fixtures verified: {len(papers) - len(problems)}/{len(papers)} (corpus {fixture_checksum()})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
