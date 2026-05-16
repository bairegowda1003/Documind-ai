"""
PDF and TXT document loader with clean text extraction.
"""

import re
import os
from pathlib import Path


def load_pdf(file_path: str) -> str:
    """Extract clean text from a PDF file page by page."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    cleaned = clean_pdf_text(text)
                    if cleaned:
                        pages_text.append(cleaned)
            except Exception:
                continue  # Skip unreadable pages

        return "\n\n".join(pages_text)

    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")


def load_txt(file_path: str) -> str:
    """Load plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read TXT: {e}")


def clean_pdf_text(text: str) -> str:
    """Remove PDF binary artifacts and normalize text."""
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove lines that are just noise (less than 3 chars)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) > 2]
    # Collapse excessive blank lines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(file_path: str) -> str:
    """Unified loader — auto-detects format."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = load_pdf(file_path)
    elif ext in (".txt", ".md"):
        text = load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not text or len(text.strip()) < 50:
        raise ValueError("Document appears to be empty or unreadable.")

    return text
