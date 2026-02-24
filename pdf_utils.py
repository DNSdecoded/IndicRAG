"""
PDF extraction and text processing utilities.
"""

import regex as re
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


def recursive_split(text: str, max_chars: int) -> List[str]:
    """Fallback recursive splitting for oversized sentences."""
    if len(text) <= max_chars:
        return [text]
    
    # Try splitting by comma
    parts = text.split(', ')
    if len(parts) > 1 and max(len(p) for p in parts) < max_chars:
        chunks = []
        current = ""
        for p in parts:
            if len(current) + len(p) + 2 > max_chars and current:
                chunks.append(current)
                current = p
            else:
                current += ", " + p if current else p
        if current:
            chunks.append(current)
        return chunks
        
    # Split by words
    words = text.split(' ')
    chunks = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > max_chars and current:
            chunks.append(current)
            current = w
        else:
            current += " " + w if current else w
    if current:
        chunks.append(current)
    return chunks

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
    
    # Protect math formulas with placeholders (inline and display forms)
    math_pattern = r'(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)'
    math_blocks = []
    
    def math_replacer(match):
        math_blocks.append(match.group(1))
        return f"__MATH_{len(math_blocks)-1}__"
    
    text = re.sub(math_pattern, math_replacer, text)
    
    # Split into sentences preserving the trailing punctuation via lookbehind.
    # - Uses (?<=[.!?]) so the punctuation stays attached to the preceding sentence.
    # - Ignores common abbreviations using a negative lookbehind for the word part.
    # - Requires the next character to be an uppercase Unicode letter (\p{Lu}).
    sentence_pattern = r'(?<=[.!?])\s+(?=\p{Lu})'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    def append_sentence_to_chunk(s: str):
        nonlocal current_chunk, chunks
        if len(current_chunk) + len(s) > max_chars and current_chunk:
            if len(current_chunk) >= config.MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + s
                else:
                    current_chunk = s
            else:
                # Chunk is too small to emit — merge into the next chunk
                # to avoid silently dropping content.
                current_chunk = current_chunk + " " + s
        else:
            current_chunk += " " + s if current_chunk else s

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(sentence) > max_chars:
            # Recursive character splitting for oversized blocks
            sub_sentences = recursive_split(sentence, max_chars)
            for sub_s in sub_sentences:
                append_sentence_to_chunk(sub_s)
        else:
            append_sentence_to_chunk(sentence)
            
    # Add the last chunk
    if current_chunk and len(current_chunk) >= config.MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
        
    # Restore math blocks
    def restore_math(chunk_text):
        for i, block in enumerate(math_blocks):
            chunk_text = chunk_text.replace(f"__MATH_{i}__", block)
        return chunk_text

    return [restore_math(c) for c in chunks]


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from text based on common section headers.
    
    Args:
        text: Full text of the paper
        
    Returns:
        List of (section_name, section_text) tuples
    """
    sections = []
    
    # Create regex pattern for section headers (must be exact lines)
    header_pattern = r'\n[ \t]*(?:\d+\.?|[A-Z]\.)?[ \t]*(' + '|'.join(config.SECTION_HEADERS) + r')[ \t]*\n'
    
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


def extract_title_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Attempt to extract the paper title from the largest font size on the first page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted title or None
    """
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return None
            page = doc[0]
            blocks = page.get_text('dict').get('blocks', [])
            candidates = []
            for b in blocks:
                for line in b.get('lines', []):
                    for span in line.get('spans', []):
                        text = span.get('text', '').strip()
                        size = span.get('size', 0)
                        if text:
                            candidates.append((size, text))
            
            candidates.sort(reverse=True, key=lambda x: x[0])
            for size, text in candidates[:10]:
                if 20 < len(text) < 200:
                    return text
    except Exception as e:
        logger.error(f"Error extracting title from PDF font info: {e}")
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
    
    if not raw_text or not raw_text.strip():
        logger.error(f"Extracted text is empty. {pdf_path} might be a scanned PDF or image. Consider using OCR.")
        return None
    
    # Clean text
    cleaned_text = clean_text(raw_text)
    
    # Extract title
    title = extract_title_from_pdf(pdf_path)
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
