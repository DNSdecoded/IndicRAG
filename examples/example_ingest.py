"""
Example: Ingest PDFs from the papers directory.
"""

import config
import ingest
from pathlib import Path


def main():
    """
    Example ingestion script.
    """
    print("=" * 60)
    print("PDF Ingestion Example")
    print("=" * 60)
    
    papers_dir = config.PAPERS_DIR
    
    # Check if papers directory has PDFs
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {papers_dir}")
        print("\nTo use this example:")
        print(f"1. Add some scientific PDF papers to: {papers_dir}")
        print("2. Run this script again")
        print("\nExample sources for open-access papers:")
        print("  - arXiv: https://arxiv.org/")
        print("  - PubMed Central: https://www.ncbi.nlm.nih.gov/pmc/")
        print("  - bioRxiv: https://www.biorxiv.org/")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files in {papers_dir}")
    print("\nOptions:")
    print("1. Ingest all PDFs (add to existing collection)")
    print("2. Reset collection and ingest all PDFs")
    print("3. Cancel")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nIngesting PDFs (adding to existing collection)...")
        stats = ingest.ingest_directory(str(papers_dir), reset=False)
    elif choice == "2":
        print("\n⚠️  This will delete all existing documents!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm == "yes":
            print("\nResetting collection and ingesting PDFs...")
            stats = ingest.ingest_directory(str(papers_dir), reset=True)
        else:
            print("Cancelled.")
            return
    else:
        print("Cancelled.")
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Successfully ingested: {stats['successful']} papers")
    print(f"Total chunks created: {stats['total_chunks']}")
    
    if stats['failed'] > 0:
        print(f"\n⚠️  Failed to ingest: {stats['failed']} papers")
    
    print("\n✅ You can now run queries using example_query.py")


if __name__ == "__main__":
    main()
