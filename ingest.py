"""
ingest.py — text extraction for all supported file types.

Supported:
  .pdf                      — text layer, OCR fallback
  .txt / .md                — plain text
  .docx / .doc               — Word documents   (requires: python-docx)
  .xlsx / .xls               — Excel sheets     (requires: openpyxl)
  .pptx / .ppt               — PowerPoint       (requires: python-pptx)
  .csv                      — comma-separated  (stdlib)
  .html / .htm               — web pages        (stdlib)

Extraction is location-aware: instead of collapsing a whole document into
one string, each extractor returns a list of (location, text) *segments*
— one per PDF page, PPTX slide, or XLSX sheet, where that concept exists
for the format. The chunker then chunks each segment independently and
tags every chunk with its segment's location, so a chunk can be cited
back as e.g. "report.pdf, p. 4" instead of just "report.pdf". Formats
with no natural segment (txt/md/docx/csv/html) get a single segment with
location=None.
"""

import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

try:
    import pymupdf as fitz  # Modern PyMuPDF import
except ImportError:
    import fitz             # Fallback for older installs

             # that pypdf sometimes returns near-empty text for (which used to
             # false-trigger the OCR fallback below on documents that actually
             # had a perfectly good text layer).

# ── tunables ──────────────────────────────────────────────────────
OCR_DPI                = 200    # was 300: Tesseract accuracy plateaus well
                                 # below 300 DPI for normal printed text, and
                                 # rendering/OCR cost scales with pixel count
                                 # (~DPI²), so this alone is a meaningful win
                                 # whenever OCR genuinely has to run.
OCR_MIN_CHARS_PER_PAGE = 20     # below this chars on a page, assume scanned → OCR
OCR_LANG               = "eng"  # tesseract lang; "eng+fra+deu" for multi-language OCR
OCR_MAX_WORKERS        = min(os.cpu_count() or 4, 8)
                                 # pytesseract shells out to the tesseract
                                 # binary and blocks on subprocess I/O, so it
                                 # releases the GIL — threads (not processes)
                                 # give real parallelism here.
CHUNK_WORDS            = 120
CHUNK_OVERLAP          = 30
MAX_CHUNKS_PER_FILE    = 4000   # safety cap so one huge file can't blow up
                                 # embedding time / memory on a single upload

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md",
    ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".csv",
    ".html", ".htm",
}


class IngestLimitExceeded(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════
# Per-format extractors — each returns list[tuple[location, text]]
# ═══════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Collapse whitespace runs and blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF ──────────────────────────────────────────────────────────
def _pdf_text_layer(path: str) -> list[str]:
    try:
        with fitz.open(path) as doc:
            return [page.get_text() or "" for page in doc]
    except Exception:
        return []


def _ocr_one_page(path: str, page_index: int) -> str:
    """Render a single page at OCR_DPI and OCR it. Runs in a worker thread —
    both convert_from_path (poppler subprocess) and pytesseract (tesseract
    subprocess) release the GIL while waiting on the external process, so
    multiple pages genuinely OCR in parallel rather than time-slicing."""
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(path, dpi=OCR_DPI, first_page=page_index + 1, last_page=page_index + 1)
    return pytesseract.image_to_string(images[0], lang=OCR_LANG) if images else ""


def _pdf_ocr_pages(path: str, page_numbers: list[int]) -> dict[int, str]:
    """OCR only the given 0-indexed pages, not the whole document, and do
    it across multiple pages at once instead of one at a time."""
    if not page_numbers:
        return {}
    workers = min(OCR_MAX_WORKERS, len(page_numbers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda i: (i, _ocr_one_page(path, i)), page_numbers)
        return dict(results)


def _extract_pdf(path: str) -> list[tuple[str, str]]:
    text_pages = _pdf_text_layer(path)
    if not text_pages:
        return []

    # Decide OCR per-page, not for the whole document — a handful of
    # image-heavy or oddly-encoded pages shouldn't drag every other page
    # (which already has a perfectly good text layer) through OCR too,
    # and we only render+OCR those specific pages, not the whole PDF.
    sparse_idx = [i for i, p in enumerate(text_pages) if len(p.strip()) < OCR_MIN_CHARS_PER_PAGE]

    pages = list(text_pages)
    if sparse_idx:
        print(f"⏳  OCR fallback for {len(sparse_idx)}/{len(text_pages)} page(s) "
              f"in '{path}' (text layer too sparse — scanned page, image-only "
              f"content, or a font the PDF parser can't decode)")
        ocr_pages = _pdf_ocr_pages(path, sparse_idx)
        for i, text in ocr_pages.items():
            pages[i] = text

    return [
        (f"p. {i}", _clean(text))
        for i, text in enumerate(pages, 1)
        if _clean(text)
    ]


# ── Plain text / Markdown ────────────────────────────────────────
def _extract_text(path: str) -> list[tuple[str, str]]:
    text = _clean(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return [(None, text)] if text else []


# ── Word / DOCX ──────────────────────────────────────────────────
def _extract_docx(path: str) -> list[tuple[str, str]]:
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
    text = _clean("\n".join(parts))
    return [(None, text)] if text else []


# ── Excel / XLSX ─────────────────────────────────────────────────
def _extract_xlsx(path: str) -> list[tuple[str, str]]:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pip install openpyxl")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    segments: list[tuple[str, str]] = []
    for ws in wb.worksheets:
        parts: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            row_text = "\t".join(cells).strip()
            if row_text:
                parts.append(row_text)
        text = _clean("\n".join(parts))
        if text:
            segments.append((f"sheet '{ws.title}'", text))
    wb.close()
    return segments


# ── PowerPoint / PPTX ────────────────────────────────────────────
def _extract_pptx(path: str) -> list[tuple[str, str]]:
    try:
        from pptx import Presentation  # python-pptx
    except ImportError:
        raise ImportError("pip install python-pptx")
    prs = Presentation(path)
    segments: list[tuple[str, str]] = []
    for i, slide in enumerate(prs.slides, 1):
        parts: list[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        text = _clean("\n".join(parts))
        if text:
            segments.append((f"slide {i}", text))
    return segments


# ── CSV ──────────────────────────────────────────────────────────
def _extract_csv(path: str) -> list[tuple[str, str]]:
    parts: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            line = "\t".join(row).strip()
            if line:
                parts.append(line)
    text = _clean("\n".join(parts))
    return [(None, text)] if text else []


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


def _extract_html(path: str) -> list[tuple[str, str]]:
    stripper = _HTMLStripper()
    stripper.feed(Path(path).read_text(encoding="utf-8", errors="ignore"))
    text = _clean("\n".join(stripper.parts))
    return [(None, text)] if text else []


# ═══════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════

def extract(path: str) -> list[tuple[str, str]]:
    """Return [(location, text), ...] segments from any supported file type."""
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
    """Sentence-aware sliding-window chunker with word-count budget.

    Strategy
    --------
    1. Split the text into sentences using punctuation boundaries
       (. ! ? followed by whitespace or end-of-string).
    2. Accumulate whole sentences until adding the next one would
       exceed `size` words.
    3. Emit the accumulated buffer as a chunk.
    4. Retain a trailing overlap: drop sentences from the *front* of
       the buffer until the retained portion is ≤ `overlap` words,
       then continue accumulating.

    This ensures no sentence is ever split across two chunks, which
    measurably improves retrieval quality compared to the old
    word-boundary-only approach.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) > int(size * 1.5):
            # Split run-on sentences or dense text without punctuation into word windows
            step = max(1, size - overlap)
            for i in range(0, len(words), step):
                chunk_s = " ".join(words[i:i + size])
                if chunk_s:
                    sentences.append(chunk_s)
        else:
            sentences.append(s)

    out: list[str] = []
    buf: list[str] = []        # sentences in current window
    buf_words: int = 0         # total word count of current window

    for sent in sentences:
        w = len(sent.split())
        # If adding this sentence would overflow the budget AND we have
        # something already, emit what we have first.
        if buf and buf_words + w > size:
            out.append(" ".join(buf))
            # Trim the front of the buffer to retain at most `overlap` words.
            while buf and buf_words > overlap:
                buf_words -= len(buf[0].split())
                buf.pop(0)

        buf.append(sent)
        buf_words += w

    # Emit any remaining sentences.
    if buf:
        out.append(" ".join(buf))

    return out



def ingest_file(path: str) -> list[dict]:
    """
    File path → list of {"text": str, "location": str | None} chunks
    ready for embedding, each tagged with where in the source document
    it came from (page/slide/sheet), when the format has that concept.
    """
    segments = extract(path)
    out: list[dict] = []
    for location, text in segments:
        for chunk in chunk_text(text):
            out.append({"text": chunk, "location": location})
            if len(out) > MAX_CHUNKS_PER_FILE:
                raise IngestLimitExceeded(
                    f"This file produced more than {MAX_CHUNKS_PER_FILE} chunks "
                    f"(very large document). Split it into smaller files and "
                    f"upload those instead."
                )
    return out
