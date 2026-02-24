#!/usr/bin/env python3
"""
Purge utility for Multilingual Scientific RAG System.
Safely clear indexed PDFs, vector database, and cached models.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Import config to get directory paths
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    # Delete entire ChromaDB directory directly
    try:
        if db_dir.exists():
            shutil.rmtree(db_dir)
            logger.info(f"Deleted database directory: {db_dir}")
            
            # Recreate empty directory
            db_dir.mkdir(exist_ok=True)
            logger.info(f"Recreated empty database directory")
    except Exception as e:
        logger.error(f"Failed to delete database directory: {e}")
        return False
    
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
        logger.info(f"Recreated empty models directory")
    except Exception as e:
        logger.error(f"Failed to delete models cache: {e}")
        return False
    
    return True


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
    
    # Check if any action requested
    if not (purge_papers_flag or purge_db_flag or purge_models_flag):
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
