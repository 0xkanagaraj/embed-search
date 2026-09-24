"""
validation.py — upload validation: size limits + magic-byte content sniffing.

The old server only checked the file *extension*. That means a renamed
.exe or a corrupt/mismatched file would sail through, get handed to the
extractor, and fail (or worse, get silently mis-parsed) deep inside
ingest.py. This module rejects those at the door, before anything is
written to disk or handed to an extractor.

No external dependency (no libmagic / python-magic) — just enough
signature-checking to catch "this clearly isn't what its extension
claims" for the file types this app actually supports.
"""

from pathlib import Path

MAX_FILE_MB   = 25          # per-file cap
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Office Open XML formats (.docx/.xlsx/.pptx) and plain .zip all start
# with the same PK signature — that's expected, we just check *a* zip
# signature is present, not which one.
_PK_ZIP = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


class ValidationError(Exception):
    pass


def check_size(size_bytes: int, filename: str) -> None:
    if size_bytes <= 0:
        raise ValidationError(f"'{filename}' is empty.")
    if size_bytes > MAX_FILE_BYTES:
        raise ValidationError(
            f"'{filename}' is {size_bytes / 1_048_576:.1f} MB, "
            f"which exceeds the {MAX_FILE_MB} MB per-file limit."
        )


def sniff_and_validate(filename: str, head: bytes) -> None:
    """
    Check that the first bytes of the upload are consistent with its
    extension. `head` should be at least the first 16 bytes of the file.
    Raises ValidationError on a mismatch; returns None if OK or if the
    type has no reliable signature to check (plain text formats).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        if not head.startswith(b"%PDF-"):
            raise ValidationError(f"'{filename}' has a .pdf extension but isn't a PDF file.")

    elif ext in (".docx", ".xlsx", ".pptx"):
        if not head.startswith(_PK_ZIP):
            raise ValidationError(
                f"'{filename}' has a .{ext.lstrip('.')} extension but isn't a valid "
                f"Office Open XML (zip-based) file."
            )

    elif ext in (".doc", ".xls", ".ppt"):
        # Legacy binary Office formats use the OLE2 compound-file signature.
        ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        if not head.startswith(ole2):
            raise ValidationError(
                f"'{filename}' has a legacy Office extension but isn't a "
                f"recognized binary Office file."
            )

    # .txt / .md / .csv / .html / .htm are free-form text — no reliable
    # magic number to check. We rely on the extension + downstream
    # extractor's own error handling for those.


def validate_upload(filename: str, content: bytes) -> None:
    """Run all checks for one uploaded file's full bytes."""
    check_size(len(content), filename)
    sniff_and_validate(filename, content[:16])
