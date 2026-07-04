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

    # Split into sentences — includes Indic terminators (purna viram, double danda).
    sentence_pattern = r'(?<=[.!?।॥])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""

    def _word_overlap(chunk_text: str) -> str:
        """Compute overlap on whole-word boundaries to avoid slicing tokens."""
        words = chunk_text.split()
        overlap_words = max(1, overlap // 6)
        tail = " ".join(words[-overlap_words:]) if len(words) > overlap_words else chunk_text
        return tail

    def append_sentence_to_chunk(s: str):
        nonlocal current_chunk, chunks
        if len(current_chunk) + len(s) > max_chars and current_chunk:
            if len(current_chunk) >= config.MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = _word_overlap(current_chunk) + " " + s
                else:
                    current_chunk = s
            else:
                current_chunk = current_chunk + " " + s
        else:
            current_chunk += " " + s if current_chunk else s

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            sub_sentences = recursive_split(sentence, max_chars)
            for sub_s in sub_sentences:
                append_sentence_to_chunk(sub_s)
        else:
            append_sentence_to_chunk(sentence)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Whitelisted headers (config.SECTION_HEADERS) as near-standalone lines, with an
# optional Arabic/Roman/letter numbering prefix.
_WHITELIST_HEADING = re.compile(
    r'(?mi)^[ \t]*(?:\d+\.?\s*|[IVXLC]{1,7}\.\s*|[A-Z]\.\s*)?('
    + '|'.join(config.SECTION_HEADERS) + r')[ \t]*$'
)

# Structural headings the whitelist misses: numbered/roman/lettered lines whose
# title isn't a known keyword — "II. THE PROPOSED FRAMEWORK", "3.2 Reward Function",
# "A. Stage 1: Global Search", "2) Improvement of F". Half the corpus uses these,
# and without them the whole method body collapses into 'introduction'.
_NUMBERED_HEADING = re.compile(
    r'(?m)^[ \t]*'
    r'(?:\d{1,2}(?:\.\d{1,2}){0,2}|[IVXLC]{1,6}|[A-Z])'  # 3 | 3.2 | II | A
    r'[.)]'                                              # . or )
    r'[ \t]+'
    r'([A-Z][^\n]{2,58})$'                               # title starting uppercase
)

# raw header keyword -> canonical bucket used for SECTION_CHUNK_SIZES lookup only
# (the raw header text is still what's stored/shown). Keeps equation-dense method
# sections on the larger chunk size so a labelled block isn't split.
_CANON = [
    (("method", "approach", "propos", "framework", "formulation", "stage",
      "surrogate", "algorithm", "network", "state", "action", "reward"), "methods"),
    (("result", "experiment", "analysis", "comparison", "validation", "performance"), "results"),
    (("discussion",), "discussion"),
    (("abstract",), "abstract"),
    (("conclusion",), "conclusion"),
]


def canonical_section(name: str) -> str:
    """Map a (possibly verbose) section header to a bucket in SECTION_CHUNK_SIZES.

    Returns the matched bucket ('methods', 'results', …) so chunk sizing applies to
    structurally-detected headers like 'stage 3: reinforcement learning'. Falls back
    to the lowercased name (→ default CHUNK_SIZE) when nothing matches.
    """
    low = name.lower()
    for keywords, bucket in _CANON:
        if any(k in low for k in keywords):
            return bucket
    return low


def _looks_like_heading(title: str) -> bool:
    """Reject numbered-list items / sentences masquerading as headings."""
    t = title.strip().rstrip('.').rstrip(':')
    if ',' in t or ';' in t:  # list items / prose
        return False
    words = t.split()
    if not (1 <= len(words) <= 8):
        return False
    alpha = [w for w in words if w[:1].isalpha()]
    if not alpha:
        return False
    caps = sum(1 for w in alpha if w[0].isupper())
    return caps >= max(1, (len(alpha) + 1) // 2)  # Title-Case / CAPS majority


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Extract (section_name, section_text) tuples via whitelist + structural headers."""
    headers = []  # (start, end, name)
    for m in _WHITELIST_HEADING.finditer(text):
        headers.append((m.start(), m.end(), m.group(1).strip().lower()))
    for m in _NUMBERED_HEADING.finditer(text):
        title = m.group(1).strip()
        if _looks_like_heading(title):
            headers.append((m.start(), m.end(), title.rstrip('.').rstrip(':').lower()))

    if not headers:
        return [("body", text)]

    headers.sort(key=lambda h: h[0])
    # Drop near-duplicate starts (whitelist + numbered both hitting one line)
    dedup = []
    for h in headers:
        if dedup and h[0] - dedup[-1][0] < 3:
            continue
        dedup.append(h)
    headers = dedup

    sections = []
    for i, (_, end, name) in enumerate(headers):
        seg_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[end:seg_end].strip()
        if body:
            sections.append((name, body))

    if headers[0][0] > 0:
        pre = text[:headers[0][0]].strip()
        if pre:
            sections.insert(0, ("header", pre))

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
                lines_info = []
                for line in b.get('lines', []):
                    spans = line.get('spans', [])
                    if not spans:
                        continue
                    # Join spans within the line — a title often splits across
                    # multiple font runs (bold/italic segments, kerning), so
                    # scoring individual spans picks a single fragment word
                    # instead of the full title text.
                    line_text = ''.join(s.get('text', '') for s in spans).strip()
                    line_size = max(s.get('size', 0) for s in spans)
                    if line_text:
                        lines_info.append((line_size, line_text))
                if not lines_info:
                    continue

                # Titles often wrap across 2+ lines at the same font size —
                # merge the leading run of same-size lines in this block
                # (stops at the first line that drops to a smaller size,
                # e.g. authors/affiliations below the title).
                max_size = max(sz for sz, _ in lines_info)
                merged = []
                for sz, text in lines_info:
                    if abs(sz - max_size) < 0.5:
                        merged.append(text)
                    else:
                        break
                candidates.append((max_size, ' '.join(merged)))

            candidates.sort(reverse=True, key=lambda x: x[0])
            for size, text in candidates[:10]:
                if 5 < len(text) < 300:
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
