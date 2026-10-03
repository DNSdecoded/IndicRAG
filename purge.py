#!/usr/bin/env python3
"""
Purge utility for Multilingual Scientific RAG System.
Safely clear indexed PDFs, vector database, and cached models.
"""

import argparse
import logging
import shutil
import sys

# Import config to get directory paths
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ChromaDB segment index directories are named after the segment UUID.
_UUID_DIR_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def confirm_action(prompt: str) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        prompt: Confirmation message to display
        
    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(f"{prompt} (y/N): ").strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no', ''):
            return False
        else:
            print("Please answer 'y' or 'n'")


def purge_papers(confirmed: bool = False) -> bool:
    """
    Delete all PDF files in the papers directory.
    
    Args:
        confirmed: If True, skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    papers_dir = config.PAPERS_DIR
    
    if not papers_dir.exists():
        logger.info(f"Papers directory does not exist: {papers_dir}")
        return True
    
    # Count PDFs
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.info(f"No PDF files found in {papers_dir}")
        return True
    
    logger.warning(f"Found {len(pdf_files)} PDF file(s) in {papers_dir}")
    
    if not confirmed:
        if not confirm_action(f"Delete all {len(pdf_files)} PDF file(s)?"):
            logger.info("Cancelled paper deletion")
            return False
    
    # Delete all PDF files
    deleted_count = 0
    for pdf_file in pdf_files:
        try:
            pdf_file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted: {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to delete {pdf_file.name}: {e}")
    
    logger.info(f"Deleted {deleted_count}/{len(pdf_files)} PDF file(s)")
    return True


def purge_database(confirmed: bool = False) -> bool:
    """
    Delete the vector database (ChromaDB persistent storage).
    
    Args:
        confirmed: If True, skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    db_dir = config.CHROMA_DB_DIR
    
    if not db_dir.exists():
        logger.info(f"Database directory does not exist: {db_dir}")
        # Create empty directory
        db_dir.mkdir(exist_ok=True)
        return True
    
    # Try to get count by reading sqlite directly without chroma to avoid persistent locks
    count = "unknown"
    try:
        import sqlite3
        db_file = db_dir / "chroma.sqlite3"
        if db_file.exists():
            conn = sqlite3.connect(db_file)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]
            finally:
                conn.close()
            logger.warning(f"Database contains {count} document chunk(s)")
    except Exception as e:
        logger.debug(f"Could not read database stats directly: {e}")
    
    if not confirmed:
        if not confirm_action(f"Delete vector database ({count} chunks)?"):
            logger.info("Cancelled database deletion")
            return False
    
    # Use ChromaDB's own client.reset() so it respects its own file locks
    # rather than deleting files out from under an active server connection.
    # NOTE: stop the API server before running this; if the server holds the
    # SQLite WAL write lock, the reset will raise an error rather than corrupt.
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        client.reset()
        logger.info("ChromaDB reset via client.reset() (collection wiped, directory kept)")
    except Exception as e:
        logger.error(f"Failed to reset database via ChromaDB client: {e}")
        logger.error("Ensure the API server is stopped before running --db purge.")
        return False

    # The vectors are gone, but the ingest log is the system of record a reindex
    # replays from and the BM25 cache is a second copy of the same corpus. Leaving
    # either behind means a purged instance still claims a corpus it cannot serve:
    # `/stats` reads zero, reindex.py rebuilds the papers the operator just purged,
    # and the next start loads a lexical index for chunks ChromaDB no longer holds.
    try:
        import persistence
        cleared = persistence.clear_ingest_log()
        logger.info("Ingest log cleared (%d event(s)) — a reindex can no longer "
                    "resurrect purged papers", cleared)
    except Exception as e:
        logger.error("Vector database was reset but the ingest log could NOT be "
                     "cleared (%s). Run reindex.py --into to rebuild, or delete "
                     "sessions.db, before ingesting again — a replay would restore "
                     "the purged corpus.", e)
        return False

    # config.BM25_CACHE_DIR, not db_dir: they happen to be the same directory
    # today, and a purge that silently stops cleaning the cache when that changes
    # is exactly the kind of coherence bug this block exists to close.
    for cache_file in config.BM25_CACHE_DIR.glob("bm25_*.json.gz"):
        try:
            cache_file.unlink()
            logger.info("Removed stale BM25 cache: %s", cache_file.name)
        except OSError as e:
            logger.warning("Could not remove BM25 cache %s: %s", cache_file.name, e)

    return True


def purge_models(confirmed: bool = False) -> bool:
    """
    Delete cached model files.
    
    Args:
        confirmed: If True, skip confirmation prompt
        
    Returns:
        True if successful, False otherwise
    """
    models_dir = config.MODELS_CACHE_DIR
    
    if not models_dir.exists():
        logger.info(f"Models cache directory does not exist: {models_dir}")
        return True
    
    # Calculate size
    total_size = 0
    file_count = 0
    for item in models_dir.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    
    if file_count == 0:
        logger.info(f"No cached models found in {models_dir}")
        return True
    
    logger.warning(f"Models cache: {file_count} file(s), {size_mb:.1f} MB")
    logger.warning("⚠️  Models will need to be re-downloaded on next use!")
    
    if not confirmed:
        if not confirm_action(f"Delete all cached models ({size_mb:.1f} MB)?"):
            logger.info("Cancelled model cache deletion")
            return False
    
    # Delete models directory
    try:
        shutil.rmtree(models_dir)
        logger.info(f"Deleted models cache directory: {models_dir}")
        
        # Recreate empty directory
        models_dir.mkdir(exist_ok=True)
        logger.info("Recreated empty models directory")
    except Exception as e:
        logger.error(f"Failed to delete models cache: {e}")
        return False
    
    return True


def purge_orphan_segments(confirmed: bool = False) -> bool:
    """Remove HNSW segment directories no live collection references.

    ChromaDB names each segment's index directory after a UUID and never garbage
    collects them: a reset or a collection recreate points the metadata at a new
    UUID and orphans the old tree on disk, whole copies of a corpus that nothing
    can read. This diffs the directory names against `segments.id` in Chroma's
    own SQLite metadata and deletes only what is unreferenced.

    Stop the API server first, same as `--db`: a running server holds these
    directories open, and Chroma is the only writer allowed to decide what is live.
    """
    import re as _re
    import sqlite3

    db_dir = config.CHROMA_DB_DIR
    meta_db = db_dir / "chroma.sqlite3"
    if not meta_db.exists():
        logger.info("No ChromaDB metadata at %s — nothing to collect", meta_db)
        return True

    try:
        conn = sqlite3.connect(f"file:{meta_db}?mode=ro", uri=True)
        try:
            referenced = {row[0] for row in conn.execute("SELECT id FROM segments")}
        finally:
            conn.close()
    except Exception as e:
        # Never guess here: an unreadable metadata DB means every directory looks
        # unreferenced, and deleting on that reading would destroy the live index.
        logger.error("Could not read segment metadata (%s) — refusing to collect", e)
        return False

    uuid_re = _re.compile(_UUID_DIR_PATTERN)
    orphans = [d for d in db_dir.iterdir()
               if d.is_dir() and uuid_re.match(d.name) and d.name not in referenced]

    if not orphans:
        logger.info("No orphaned segment directories (%d referenced)", len(referenced))
        return True

    total = sum(f.stat().st_size for d in orphans for f in d.rglob("*") if f.is_file())
    logger.info("Found %d orphaned segment director(ies), %.1f MB, "
                "against %d referenced by live collections",
                len(orphans), total / (1024 * 1024), len(referenced))
    for d in orphans:
        logger.info("  %s", d.name)

    if not confirmed:
        if not confirm_action(f"Delete {len(orphans)} orphaned segment director(ies)?"):
            logger.info("Cancelled segment collection")
            return False

    ok = True
    for d in orphans:
        try:
            shutil.rmtree(d)
            logger.info("Removed %s", d.name)
        except OSError as e:
            logger.error("Could not remove %s: %s (is the API server running?)", d.name, e)
            ok = False
    return ok


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Purge indexed data from Multilingual RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python purge.py --papers          # Delete all PDFs
  python purge.py --db              # Clear vector database
  python purge.py --all --yes       # Delete everything without prompts
  python purge.py --db --papers     # Delete PDFs and database
  python purge.py --segments        # Reclaim orphaned HNSW segment dirs
        """
    )
    
    parser.add_argument(
        '--papers',
        action='store_true',
        help='Delete all PDF files in papers/ directory'
    )
    parser.add_argument(
        '--db',
        action='store_true',
        help='Delete vector database (ChromaDB data)'
    )
    parser.add_argument(
        '--models',
        action='store_true',
        help='Delete cached models (they will be re-downloaded on next use)'
    )
    parser.add_argument(
        '--segments',
        action='store_true',
        help='Delete orphaned ChromaDB HNSW segment directories (safe: keeps everything '
             'a live collection references). Not included in --all.'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Delete papers, database, and models'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Auto-confirm all prompts (non-interactive mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine what to purge
    purge_papers_flag = args.papers or args.all
    purge_db_flag = args.db or args.all
    purge_models_flag = args.models or args.all
    # Deliberately NOT part of --all: --all is the destructive sweep, while this
    # only reclaims disk nothing references, and it is the one operation an
    # operator may want to run on a corpus they intend to keep.
    purge_segments_flag = args.segments
    
    # Check if any action requested
    if not (purge_papers_flag or purge_db_flag or purge_models_flag
            or purge_segments_flag):
        parser.print_help()
        logger.error("\nError: No action specified. Use --papers, --db, --models, or --all")
        sys.exit(1)
    
    # Show what will be done
    logger.info("=" * 60)
    logger.info("RAG System Purge Utility")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Actions to perform:")
    if purge_papers_flag:
        logger.info("  ✓ Delete PDF files in papers/")
    if purge_db_flag:
        logger.info("  ✓ Delete vector database")
    if purge_models_flag:
        logger.info("  ✓ Delete cached models")
    if purge_segments_flag:
        logger.info("  ✓ Delete orphaned ChromaDB segment directories")
    logger.info("")
    
    # Confirm if not auto-yes
    if not args.yes:
        if not confirm_action("Proceed with purge?"):
            logger.info("Purge cancelled")
            sys.exit(0)
    
    # Perform purge operations
    success = True
    
    if purge_papers_flag:
        logger.info("\n" + "-" * 60)
        logger.info("Purging papers...")
        logger.info("-" * 60)
        if not purge_papers(confirmed=args.yes):
            success = False
    
    if purge_db_flag:
        logger.info("\n" + "-" * 60)
        logger.info("Purging database...")
        logger.info("-" * 60)
        if not purge_database(confirmed=args.yes):
            success = False
    
    if purge_models_flag:
        logger.info("\n" + "-" * 60)
        logger.info("Purging model cache...")
        logger.info("-" * 60)
        if not purge_models(confirmed=args.yes):
            success = False

    if purge_segments_flag:
        logger.info("-" * 60)
        logger.info("Collecting orphaned segment directories...")
        logger.info("-" * 60)
        if not purge_orphan_segments(confirmed=args.yes):
            success = False
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✓ Purge completed successfully")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("✗ Purge completed with errors")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
