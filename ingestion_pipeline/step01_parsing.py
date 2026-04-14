import os
from importlib import import_module


def _get_pdf_reader_class():
    try:
        module = import_module("pypdf")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    return module.PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    PdfReader = _get_pdf_reader_class()
    reader = PdfReader(pdf_path)
    pages_text: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)

    return "\n\n".join(pages_text)


def extract_text_by_page(pdf_path: str) -> list[dict[str, str | int]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    PdfReader = _get_pdf_reader_class()
    reader = PdfReader(pdf_path)

    pages: list[dict[str, str | int]] = []
    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        pages.append(
            {
                "page_number": page_index + 1,
                "text": text,
            }
        )

    return pages
