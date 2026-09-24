"""
ingest.py — text extraction for all supported file types.

Supported:
  .pdf                     — text layer, OCR fallback
  .txt / .md               — plain text
  .docx / .doc             — Word documents   (requires: python-docx)
  .xlsx / .xls             — Excel sheets     (requires: openpyxl)
  .pptx / .ppt             — PowerPoint       (requires: python-pptx)
  .csv                     — comma-separated  (stdlib)
  .html / .htm             — web pages        (stdlib)

All extractors strip noise (empty lines, header/footer artefacts) and
return a single cleaned string that the chunker then splits.
"""

import csv
import re
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

# ── tunables ──────────────────────────────────────────────────────
OCR_DPI      = 300
OCR_MIN_CHARS = 50
OCR_LANG     = "eng"     # tesseract lang; "eng+fra+deu" for multi-language OCR
CHUNK_WORDS  = 120
CHUNK_OVERLAP = 30

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md",
    ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".csv",
    ".html", ".htm",
}


# ═══════════════════════════════════════════════════════════════════
# Per-format extractors
# ═══════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Collapse whitespace runs and blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF ──────────────────────────────────────────────────────────
def _pdf_text_layer(path: str) -> list[str]:
    try:
        reader = PdfReader(path)
        return [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return []


def _pdf_ocr(path: str) -> list[str]:
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(path, dpi=OCR_DPI)
    return [pytesseract.image_to_string(img, lang=OCR_LANG) for img in images]


def _extract_pdf(path: str) -> str:
    pages = _pdf_text_layer(path)
    if sum(len(p.strip()) for p in pages) >= OCR_MIN_CHARS:
        return _clean("\n\n".join(pages))
    return _clean("\n\n".join(_pdf_ocr(path)))


# ── Plain text / Markdown ────────────────────────────────────────
def _extract_text(path: str) -> str:
    return _clean(Path(path).read_text(encoding="utf-8", errors="ignore"))


# ── Word / DOCX ──────────────────────────────────────────────────
def _extract_docx(path: str) -> str:
    try:
        from docx import Document  # python-docx
    except ImportError:
        raise ImportError("pip install python-docx")
    doc = Document(path)
    parts: list[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if cells:
                parts.append(cells)
    return _clean("\n".join(parts))


# ── Excel / XLSX ─────────────────────────────────────────────────
def _extract_xlsx(path: str) -> str:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pip install openpyxl")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        parts.append(f"[Sheet: {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            row_text = "\t".join(cells).strip()
            if row_text:
                parts.append(row_text)
    wb.close()
    return _clean("\n".join(parts))


# ── PowerPoint / PPTX ────────────────────────────────────────────
def _extract_pptx(path: str) -> str:
    try:
        from pptx import Presentation  # python-pptx
    except ImportError:
        raise ImportError("pip install python-pptx")
    prs = Presentation(path)
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"[Slide {i}]")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
    return _clean("\n".join(parts))


# ── CSV ──────────────────────────────────────────────────────────
def _extract_csv(path: str) -> str:
    parts: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            line = "\t".join(row).strip()
            if line:
                parts.append(line)
    return _clean("\n".join(parts))


# ── HTML / HTM ───────────────────────────────────────────────────
class _HTMLStripper(HTMLParser):
    _SKIP_TAGS = {"script", "style", "head", "meta", "link",
                  "noscript", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def _extract_html(path: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return _clean("\n".join(stripper.parts))


# ═══════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════

def extract(path: str) -> str:
    """Return raw text from any supported file type."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".txt", ".md"):
        return _extract_text(path)
    if ext in (".docx", ".doc"):
        return _extract_docx(path)
    if ext in (".xlsx", ".xls"):
        return _extract_xlsx(path)
    if ext in (".pptx", ".ppt"):
        return _extract_pptx(path)
    if ext == ".csv":
        return _extract_csv(path)
    if ext in (".html", ".htm"):
        return _extract_html(path)
    raise ValueError(f"Unsupported file type: '{ext}'. "
                     f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


# ═══════════════════════════════════════════════════════════════════
# Chunker
# ═══════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Word-based sliding window chunker with overlap."""
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + size])
        if chunk.strip():
            out.append(chunk)
        i += size - overlap
    return out


def ingest_file(path: str) -> list[str]:
    """File path → list of text chunks ready for embedding."""
    text = extract(path)
    return chunk_text(text)
