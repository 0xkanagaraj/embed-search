import pytest

import validation


def test_rejects_empty_file():
    with pytest.raises(validation.ValidationError):
        validation.check_size(0, "x.txt")


def test_rejects_oversize_file():
    with pytest.raises(validation.ValidationError):
        validation.check_size(validation.MAX_FILE_BYTES + 1, "big.pdf")


def test_accepts_normal_size():
    validation.check_size(1000, "ok.txt")  # should not raise


def test_valid_pdf_header_passes():
    validation.sniff_and_validate("a.pdf", b"%PDF-1.4\n%rest")


def test_fake_pdf_rejected():
    with pytest.raises(validation.ValidationError):
        validation.sniff_and_validate("fake.pdf", b"this is not a pdf")


def test_valid_docx_zip_header_passes():
    validation.sniff_and_validate("a.docx", b"PK\x03\x04" + b"\x00" * 12)


def test_fake_docx_rejected():
    with pytest.raises(validation.ValidationError):
        validation.sniff_and_validate("fake.docx", b"not a zip header!")


def test_plain_text_has_no_signature_check():
    # .txt has no magic number — should never raise regardless of content
    validation.sniff_and_validate("plain.txt", b"anything at all here")


def test_validate_upload_combines_size_and_sniff(tmp_path):
    with pytest.raises(validation.ValidationError):
        validation.validate_upload("fake.pdf", b"not a real pdf")
    validation.validate_upload("real.pdf", b"%PDF-1.4" + b"\x00" * 100)
