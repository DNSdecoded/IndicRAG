"""
PDF extraction and text processing utilities.
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import config

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            return text
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise, normalizing whitespace, etc.
    
    Important: Preserves newlines for title/section extraction.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    # Remove noise patterns (page numbers, copyright, etc.)
    for pattern in config.NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize Windows line endings to Unix
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Collapse multiple spaces and tabs (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove very long sequences of dots or dashes (often from TOC)
    text = re.sub(r'[.\-_]{4,}', '', text)
    
    # Remove standalone numbers at line boundaries (page numbers)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Clean up excessive newlines (3+ consecutive → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def simple_chunk(text: str, max_chars: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk (default from config)
        overlap: Overlap between chunks in characters (default from config)
        
    Returns:
        List of text chunks
    """
    if max_chars is None:
        max_chars = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    
    # Split into sentences using robust pattern that handles scientific abbreviations
    # This pattern:
    # - Looks for sentence endings (.!?) followed by whitespace
    # - Ignores common abbreviations (Fig., Eq., Dr., Mr., Ms., Prof., vs., etc., al., approx.)
    # - Requires next character to be uppercase (indicating new sentence)
    # - Handles edge cases like "1.0" by not splitting after digits
    sentence_pattern = r'(?<![A-Z])(?<!\b[Ff]ig)(?<!\b[Ee]q)(?<!\b[Dd]r)(?<!\b[Mm]r)(?<!\b[Mm]s)(?<!\b[Pp]rof)(?<!\bvs)(?<!\betc)(?<!\bal)(?<!\bapprox)(?<!\d)[.!?]\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chars, save current chunk
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            if len(current_chunk) >= config.MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk and len(current_chunk) >= config.MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from text based on common section headers.
    
    Args:
        text: Full text of the paper
        
    Returns:
        List of (section_name, section_text) tuples
    """
    sections = []
    
    # Create regex pattern for section headers
    header_pattern = r'\n\s*(' + '|'.join(config.SECTION_HEADERS) + r')\s*\n'
    
    # Find all section headers
    matches = list(re.finditer(header_pattern, text, re.IGNORECASE))
    
    if not matches:
        # No sections found, return entire text as "body"
        return [("body", text)]
    
    # Extract sections
    for i, match in enumerate(matches):
        section_name = match.group(1).lower()
        start_pos = match.end()
        
        # End position is the start of next section or end of text
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        section_text = text[start_pos:end_pos].strip()
        
        if section_text:
            sections.append((section_name, section_text))
    
    # If there's text before the first section, add it as "header"
    if matches[0].start() > 0:
        header_text = text[:matches[0].start()].strip()
        if header_text:
            sections.insert(0, ("header", header_text))
    
    return sections


def extract_title_from_text(text: str) -> Optional[str]:
    """
    Attempt to extract the paper title from the beginning of the text.
    
    Args:
        text: Full text of the paper
        
    Returns:
        Extracted title or None
    """
    # Take first few lines
    lines = text.split('\n')[:10]
    
    # Look for a line that looks like a title (longer than 20 chars, not all caps)
    for line in lines:
        line = line.strip()
        if 20 < len(line) < 200 and not line.isupper():
            # Check if it's not a common header
            if not any(header in line.lower() for header in ['abstract', 'introduction', 'arxiv']):
                return line
    
    return None


def process_pdf(pdf_path: str) -> Dict:
    """
    Process a PDF file: extract text, clean, detect sections.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with:
            - 'path': original path
            - 'title': extracted title
            - 'text': cleaned full text
            - 'sections': list of (section_name, section_text) tuples
    """
    # Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        return None
    
    # Clean text
    cleaned_text = clean_text(raw_text)
    
    # Extract title
    title = extract_title_from_text(cleaned_text)
    if not title:
        # Use filename as fallback
        title = Path(pdf_path).stem
    
    # Extract sections
    sections = extract_sections(cleaned_text)
    
    return {
        'path': pdf_path,
        'title': title,
        'text': cleaned_text,
        'sections': sections
    }


if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        logger.info(f"Processing: {pdf_path}")
        logger.info("-" * 60)
        
        result = process_pdf(pdf_path)
        
        if result:
            logger.info(f"Title: {result['title']}")
            logger.info(f"Total text length: {len(result['text'])} characters")
            logger.info(f"\nSections found: {len(result['sections'])}")
            for section_name, section_text in result['sections']:
                logger.info(f"  - {section_name}: {len(section_text)} chars")
            
            # Test chunking
            chunks = simple_chunk(result['text'])
            logger.info(f"\nChunks created: {len(chunks)}")
            logger.info(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
        else:
            logger.error("Failed to process PDF")
    else:
        logger.info("Usage: python pdf_utils.py <path_to_pdf>")
