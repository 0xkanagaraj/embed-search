import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz


OCR_DPI                = 200
OCR_MIN_CHARS_PER_PAGE = 20
OCR_LANG               = "eng"
OCR_MAX_WORKERS        = min(os.cpu_count() or 4, 8)  # threads are enough: OCR waits on subprocesses
CHUNK_WORDS            = 120
CHUNK_OVERLAP          = 30
MAX_CHUNKS_PER_FILE    = 4000

# Legacy .doc/.xls/.ppt are unsupported: the parsing libraries only read the zip-based formats.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md",
    ".docx", ".xlsx", ".pptx",
    ".csv",
    ".html", ".htm",
}


class IngestLimitExceeded(Exception):
    pass


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_text_layer(path: str) -> list[str]:
    with fitz.open(path) as doc:
        return [page.get_text() or "" for page in doc]


def _ocr_one_page(path: str, page_index: int) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(path, dpi=OCR_DPI, first_page=page_index + 1, last_page=page_index + 1)
        return pytesseract.image_to_string(images[0], lang=OCR_LANG) if images else ""
    except Exception as e:
        print(f"⚠️  OCR failed on page {page_index + 1} of '{path}': {e}")
        return ""


def _pdf_ocr_pages(path: str, page_numbers: list[int]) -> dict[int, str]:
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


def _extract_text(path: str) -> list[tuple[str, str]]:
    text = _clean(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return [(None, text)] if text else []


def _extract_docx(path: str) -> list[tuple[str, str]]:
    try:
        from docx import Document
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


def _extract_xlsx(path: str) -> list[tuple[str, str]]:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pip install openpyxl")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    segments: list[tuple[str, str]] = []
    try:
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
    finally:
        wb.close()
    return segments


def _extract_pptx(path: str) -> list[tuple[str, str]]:
    try:
        from pptx import Presentation
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


class _HTMLStripper(HTMLParser):
    # Paired tags only: void tags like <meta>/<link> never close and would leave the skip counter stuck.
    _SKIP_TAGS = {"script", "style", "head", "noscript", "nav", "footer", "header"}

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


def extract(path: str) -> list[tuple[str, str]]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".txt", ".md"):
        return _extract_text(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".xlsx":
        return _extract_xlsx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext == ".csv":
        return _extract_csv(path)
    if ext in (".html", ".htm"):
        return _extract_html(path)
    raise ValueError(f"Unsupported file type: '{ext}'. "
                     f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def chunk_text(
    text: str,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if len(words) > int(size * 1.5):
            step = max(1, size - overlap)
            for i in range(0, len(words), step):
                chunk_s = " ".join(words[i:i + size])
                if chunk_s:
                    sentences.append(chunk_s)
        else:
            sentences.append(s)

    out: list[str] = []
    buf: list[str] = []
    buf_words: int = 0

    for sent in sentences:
        w = len(sent.split())
        if buf and buf_words + w > size:
            out.append(" ".join(buf))
            while buf and buf_words > overlap:
                buf_words -= len(buf[0].split())
                buf.pop(0)

        buf.append(sent)
        buf_words += w

    if buf:
        out.append(" ".join(buf))

    return out


def ingest_file(path: str) -> list[dict]:
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
