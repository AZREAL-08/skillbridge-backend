import io
import logging
import re
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Strict Security Constraints
MAX_PDF_PAGES = 10
SUSPICIOUS_PATTERNS = [
    rb"/JS", rb"/JavaScript", rb"/AA", rb"/OpenAction", rb"/RichMedia", rb"/Launch"
]

def validate_pdf_safety(file_bytes: bytes):
    """
    Performs strict file checking for potential malware or suspicious elements.
    """
    # 1. Magic Number Check (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid file format: Not a valid PDF magic number.")

    # 2. Check for suspicious PDF objects (Active content like JS)
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in file_bytes:
            logger.warning(f"[Security] Suspicious pattern {pattern} found in uploaded PDF.")
            raise ValueError("Security violation: Suspicious active content detected in PDF.")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts raw text from a PDF file with safety checks and limits.
    """
    validate_pdf_safety(file_bytes)
    
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        
        # 3. Page Count Limit
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"File too large: Resume exceeds maximum allowed pages ({MAX_PDF_PAGES}).")

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Clean up common PDF artifacts
        text = text.replace('\x00', '') # Remove null bytes
        
        if not text.strip():
            logger.warning("[PDFUtils] No text could be extracted from the PDF.")
            
        return text.strip()
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[PDFUtils] Failed to extract text from PDF: {e}", exc_info=True)
        raise ValueError(f"Could not parse PDF file: {str(e)}")
