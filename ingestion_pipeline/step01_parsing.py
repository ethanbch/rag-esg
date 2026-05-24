import os
from importlib import import_module
from typing import Any


def _get_pymupdf4llm_module() -> Any:
    try:
        # pymupdf4llm est le standard de l'industrie pour le RAG sur des PDF complexes
        module = import_module("pymupdf4llm")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'pymupdf4llm'. Install it with: pip install pymupdf4llm"
        ) from exc
    return module


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrait le contenu d'un PDF au format Markdown.
    Idéal pour conserver la structure des tableaux et des doubles colonnes pour le LLM.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pymupdf4llm = _get_pymupdf4llm_module()

    # to_markdown convertit intelligemment le PDF (y compris les tableaux) en Markdown pur
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Sauvegarde du markdown généré pour analyse ou inspection
    md_path = f"{os.path.splitext(pdf_path)[0]}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return md_text


def extract_text_by_page(pdf_path: str) -> list[dict[str, str | int]]:
    """
    Extrait le contenu page par page au format Markdown, en préservant
    les numéros de pages pour les citations (idéal pour l'UI Streamlit).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pymupdf4llm = _get_pymupdf4llm_module()

    # L'option page_chunks=True retourne une liste de dictionnaires (un par page)
    # contenant le texte Markdown et les métadonnées extraites.
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    pages: list[dict[str, str | int]] = []
    for index, chunk in enumerate(chunks):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        # pymupdf4llm stocke le numéro de page (0-indexé) dans les métadonnées
        page_num = chunk.get("metadata", {}).get("page", index) + 1

        pages.append(
            {
                "page_number": page_num,
                "text": text,
            }
        )

    return pages
