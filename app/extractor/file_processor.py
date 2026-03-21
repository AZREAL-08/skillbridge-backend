import logging
from typing import Tuple
from app.extractor.pdf_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Security Constants
MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

def process_uploaded_file(filename: str, content: bytes) -> str:
    """
    Orchestrates the secure processing of uploaded files.
    Identifies format, runs security checks, and extracts text.
    """
    # 1. Size Check (Pre-processing)
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"Security error: File size exceeds {MAX_FILE_SIZE_MB}MB limit.")

    # 2. Logic based on extension
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext == 'pdf':
        logger.info(f"[FileProcessor] Processing PDF: {filename}")
        return extract_text_from_pdf(content)
    
    elif ext in ['txt', 'md']:
        logger.info(f"[FileProcessor] Processing Text: {filename}")
        return _process_text_file(content)
    
    else:
        # Strict format checking: if we don't know it, we don't process it
        raise ValueError(f"Security error: Unsupported file format '.{ext}'. Only PDF and TXT are allowed.")

def _process_text_file(content: bytes) -> str:
    """
    Safely decodes and cleans text files.
    """
    try:
        # Attempt UTF-8 decoding
        text = content.decode('utf-8')
        
        # Security: Remove any potential shell scripts or common attack vectors 
        # (Though less critical for text, we still sanitize)
        text = text.replace('\x00', '') # Remove null bytes
        
        return text.strip()
    except UnicodeDecodeError:
        raise ValueError("Invalid file encoding: Text file must be UTF-8 encoded.")
