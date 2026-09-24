from pathlib import Path
from pypdf import PdfReader

OCR_DPI = 300
OCR_MIN_CHARS = 50
OCR_LANG = "eng"

CHUNK_SIZE = 120
CHUNK_OVERLAP = 30


def _extract_text_layer(path: str) -> list[str]:
    try:
        reader = PdfReader(path)
        return [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return []


def _extract_via_ocr(path: str) -> list[str]:
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(path, dpi=OCR_DPI)
    return [pytesseract.image_to_string(img, lang=OCR_LANG) for img in images]


def load_pages(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        return [p.read_text(encoding="utf-8", errors="ignore")]
    pages = _extract_text_layer(path)
    if sum(len(x.strip()) for x in pages) >= OCR_MIN_CHARS:
        return pages
    return _extract_via_ocr(path)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    out, i = [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in out if c.strip()]


def ingest_file(path: str) -> list[str]:
    """File path → list of text chunks."""
    pages = load_pages(path)
    return chunk_text("\n\n".join(pages))