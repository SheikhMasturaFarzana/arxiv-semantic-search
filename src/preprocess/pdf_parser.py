from __future__ import annotations

import re
import hashlib
from pathlib import Path
from typing import Optional

import requests
import fitz  # PyMuPDF

PDF_CACHE_DIR = Path("data/pdf_cache")
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_arxiv_id_from_url(url_pdf: str) -> str:
    """
    Extract arXiv ID from a standard arXiv PDF URL.
    Examples:
      https://arxiv.org/pdf/2401.12345.pdf   -> 2401.12345
      https://arxiv.org/pdf/2401.12345v2.pdf -> 2401.12345v2
    Fallback: short hash if parsing fails.
    """
    m = re.search(r"/pdf/([0-9]{4}\.[0-9]{4,5}(v\\d+)?)\\.pdf$", url_pdf)
    if m:
        return m.group(1)
    return hashlib.md5(url_pdf.encode("utf-8")).hexdigest()[:12]


def _cached_pdf_path(url_pdf: str) -> Path:
    """Return the local cache path for a PDF URL."""
    return PDF_CACHE_DIR / f"{_extract_arxiv_id_from_url(url_pdf)}.pdf"


def download_pdf(url_pdf: str, *, force: bool = False, timeout: int = 60) -> Optional[Path]:
    """
    Download the arXiv PDF to cache if missing (or force=True).
    Enforces a hard timeout cap. If download fails, returns None.
    """
    if not url_pdf or not url_pdf.startswith("http"):
        raise ValueError("A valid arXiv PDF URL is required.")

    path = _cached_pdf_path(url_pdf)
    if path.exists() and not force:
        return path

    headers = {"User-Agent": "arxiv-semantic-search/1.0"}
    try:
        # timeout=(connect, read). Example: (10, 30)
        with requests.get(url_pdf, headers=headers, stream=True, timeout=(10, 30)) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
        return path
    except Exception as e:
        print(f"[warn] Failed to download PDF: {url_pdf} ({e})")
        return None


def extract_first_page_text_from_pdf(pdf_path: Path) -> str:
    """
    Open the PDF with PyMuPDF and return plain text of the first page.
    Returns "" if empty or invalid.
    """
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                return ""
            page = doc.load_page(0)
            return (page.get_text("text") or "").strip()
    except Exception as e:
        print(f"[warn] Failed to read PDF file {pdf_path}: {e}")
        return ""


def extract_first_page_text(url_pdf: str, *, force_redownload: bool = False) -> str:
    """
    High-level helper:
      1) download (or reuse cached) PDF
      2) extract first-page text with PyMuPDF
    Returns:
      str (first-page text, possibly empty)
    """
    pdf_path = download_pdf(url_pdf, force=force_redownload)
    if not pdf_path or not pdf_path.exists():
        return ""  # gracefully skip if download failed
    return extract_first_page_text_from_pdf(pdf_path)