#!/usr/bin/env python3
"""Backup and restore for IndicRAG.

What gets backed up is the ingest log, not the vector store. The log is the
system of record; ChromaDB's HNSW segments and the BM25 index are derived views
that a replay reproduces exactly (see reindex.py). So a snapshot is one small
SQLite file instead of gigabytes of index, it can be taken with the server
running, and restoring it cannot leave a half-copied index behind — the worst
case is a rebuild.

The snapshot also carries sessions, jobs, watches, reports and feedback, since
they live in the same database and are NOT derivable from anything.

Usage
-----
    python backup.py create                  # snapshot into backups/
    python backup.py list                    # what is on disk
    python backup.py restore <file> --yes    # replace the log, then rebuild
    python backup.py restore <file> --yes --no-rebuild
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backup")

BACKUP_DIR = Path(config.PROJECT_ROOT) / "backups"


def _manifest_path(snapshot: Path) -> Path:
    return snapshot.with_suffix(".json")


def create(out_dir=None) -> Path:
    """Snapshot the database, with a manifest describing what it holds."""
    import persistence

    out_dir = Path(out_dir or BACKUP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, not just a timestamp: the stamp has second resolution,
    # so two snapshots started in the same second would silently overwrite each
    # other — and a backup that quietly replaces another backup is worse than no
    # backup at all.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = out_dir / f"indicrag-{stamp}.db"
    suffix = 1
    while True:
        try:
            snapshot.touch(exist_ok=False)
            break
        except FileExistsError:
            snapshot = out_dir / f"indicrag-{stamp}-{suffix}.db"
            suffix += 1

    persistence.snapshot_to(snapshot)

    events = persistence.get_ingest_events()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "papers": len(events),
        "chunks": sum(len(e["chunks"]) for e in events),
        # Which weights produced the vectors this log describes. A restore onto a
        # build with a different embedding model is legal — the replay re-embeds —
        # but the operator should know that is what is happening.
        "embed_models": sorted({e["embed_model"] for e in events if e["embed_model"]}),
        "collection": config.COLLECTION_NAME,
        "version": getattr(config, "VERSION", "unknown"),
    }
    _manifest_path(snapshot).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size_mb = snapshot.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB): %d papers / %d chunks",
                snapshot, size_mb, manifest["papers"], manifest["chunks"])
    return snapshot


def list_backups(out_dir=None) -> list:
    out_dir = Path(out_dir or BACKUP_DIR)
    if not out_dir.exists():
        logger.info("No backup directory at %s", out_dir)
        return []
    snapshots = sorted(out_dir.glob("indicrag-*.db"))
    if not snapshots:
        logger.info("No snapshots in %s", out_dir)
        return []
    for snap in snapshots:
        try:
            m = json.loads(_manifest_path(snap).read_text(encoding="utf-8"))
            detail = (f"{m['papers']} papers / {m['chunks']} chunks, "
                      f"collection={m['collection']}, version={m['version']}")
        except Exception:
            detail = "no manifest"
        logger.info("  %-32s %6.1f MB  %s", snap.name,
                    snap.stat().st_size / (1024 * 1024), detail)
    return snapshots


def restore(snapshot, rebuild: bool = True, into: str = None,
            confirmed: bool = False) -> int:
    """Replace the live database with `snapshot`, then replay it into the indexes.

    The replay is the point: restoring the log alone would leave the vector store
    describing the corpus from before the restore, which is a worse state than
    either snapshot — retrieval would answer from chunks the log no longer knows.
    """
    import persistence

    snapshot = Path(snapshot)
    if not snapshot.exists():
        logger.error("No such snapshot: %s", snapshot)
        return 2

    if not confirmed:
        # Not a prompt-if-tty nicety: this overwrites the system of record.
        logger.error("Refusing to overwrite %s without --yes. Take a fresh "
                     "snapshot first (python backup.py create).",
                     config.SESSIONS_DB_PATH)
        return 2

    try:
        result = persistence.restore_from(snapshot)
    except Exception as e:
        logger.error("Restore failed, the existing database is unchanged: %s", e)
        return 2
    logger.info("Restored %d ingest event(s) from %s", result["papers"], snapshot)

    if not rebuild:
        logger.warning("Indexes NOT rebuilt. They still describe the pre-restore "
                       "corpus — run reindex.py before serving queries.")
        return 0

    import reindex

    target = into or config.COLLECTION_NAME
    logger.info("Replaying the restored log into '%s'", target)
    # confirm=True: the operator already said --yes to the destructive half of
    # this, and a restore that stops short of rebuilding is the broken state.
    return reindex.reindex(target, dry_run=False, batch_size=64, confirm=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="write a snapshot of the database")
    p_create.add_argument("--out", default=None, help=f"directory (default: {BACKUP_DIR})")

    p_list = sub.add_parser("list", help="list snapshots on disk")
    p_list.add_argument("--out", default=None, help=f"directory (default: {BACKUP_DIR})")

    p_restore = sub.add_parser("restore", help="restore a snapshot and rebuild the indexes")
    p_restore.add_argument("snapshot")
    p_restore.add_argument("--no-rebuild", action="store_true",
                           help="restore the log only, leaving the indexes stale")
    p_restore.add_argument("--into", default=None,
                           help="replay into this collection instead of the live one")
    p_restore.add_argument("--yes", action="store_true",
                           help="confirm overwriting the live database")

    args = ap.parse_args()

    if args.command == "create":
        create(args.out)
        return 0
    if args.command == "list":
        list_backups(args.out)
        return 0
    return restore(args.snapshot, rebuild=not args.no_rebuild,
                   into=args.into, confirmed=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
