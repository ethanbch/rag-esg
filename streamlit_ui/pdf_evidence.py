from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import streamlit as st

from .state import (
    _build_highlight_query,
    _resolve_pdf_source_for_chunks,
    _select_primary_pdf_filename,
)


def _search_rects_with_fallback(page: Any, query: str) -> list[Any]:
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        return []

    words = normalized.split()
    candidates = [normalized]
    for count in (20, 14, 10, 6):
        if len(words) >= count:
            candidates.append(" ".join(words[:count]))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        rects = page.search_for(candidate)
        if rects:
            return rects

    return []


def _map_queries_to_pages(doc: Any, queries: list[str]) -> dict[int, list[str]]:
    unique_queries: list[str] = []
    seen_queries: set[str] = set()

    for query in queries:
        normalized_query = re.sub(r"\s+", " ", str(query)).strip()
        if not normalized_query or normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        unique_queries.append(normalized_query)

    if not unique_queries:
        return {}

    page_count = int(getattr(doc, "page_count", 0))
    max_pages_to_scan = min(page_count, 120)
    if max_pages_to_scan <= 0:
        return {}

    mapped_page_queries: dict[int, list[str]] = defaultdict(list)
    for query in unique_queries[:12]:
        for page_index in range(max_pages_to_scan):
            page = doc[page_index]
            rects = _search_rects_with_fallback(page, query)
            if not rects:
                continue

            mapped_page_queries[page_index + 1].append(query)
            break

    return mapped_page_queries


def _render_pdf_evidence_for_chunks(
    chunks: list[dict[str, Any]],
    evidence_key: str | None = None,
    auto_render: bool = False,
) -> None:
    if not chunks:
        return

    if evidence_key is None:
        seed = "|".join(str(chunk.get("chunk_id", "")) for chunk in chunks[:6])
        if not seed:
            seed = str(len(chunks))
        evidence_key = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]

    visible_state_key = f"pdf_evidence_visible_{evidence_key}"
    is_visible = bool(st.session_state.get(visible_state_key, False))

    pdf_source = _resolve_pdf_source_for_chunks(chunks)
    if pdf_source is None:
        primary_pdf_name = _select_primary_pdf_filename(chunks)
        if primary_pdf_name:
            with st.expander("PDF evidence", expanded=False):
                st.info(
                    f"Source PDF not found locally for highlighting: {primary_pdf_name}"
                )
        return

    filename = str(pdf_source.get("filename", "source.pdf"))
    pdf_bytes_raw = pdf_source.get("pdf_bytes")
    pdf_path_raw = pdf_source.get("pdf_path")

    pdf_bytes: bytes | None = (
        bytes(pdf_bytes_raw)
        if isinstance(pdf_bytes_raw, (bytes, bytearray)) and pdf_bytes_raw
        else None
    )
    pdf_path = str(pdf_path_raw).strip() if isinstance(pdf_path_raw, str) else ""

    if pdf_bytes is None and not pdf_path:
        return

    try:
        import fitz  # type: ignore
    except Exception:
        with st.expander("PDF evidence", expanded=False):
            st.info("Install pymupdf to display highlighted PDF evidence.")
        return

    page_queries: dict[int, list[str]] = defaultdict(list)
    unscoped_queries: list[str] = []
    for chunk in chunks:
        metadata = (
            chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        )
        query = str(metadata.get("highlight_query", "")).strip()
        if not query:
            query = _build_highlight_query(str(chunk.get("content", "")))
        if not query:
            continue

        page_number_raw = metadata.get("page_number")
        try:
            page_number = int(page_number_raw)
        except (TypeError, ValueError):
            page_number = None

        if isinstance(page_number, int) and page_number > 0:
            page_queries[page_number].append(query)
        else:
            unscoped_queries.append(query)

    if not page_queries and not unscoped_queries:
        return

    with st.expander("PDF evidence (highlighted)", expanded=False):
        st.caption("Highlighted passages come from retrieved chunks for this answer.")

        if not auto_render and not is_visible:
            if st.button(
                "Show highlighted PDF evidence",
                key=f"show_pdf_evidence_{evidence_key}",
                width="stretch",
            ):
                st.session_state[visible_state_key] = True
                st.rerun()
            else:
                st.info(
                    "Click the button to generate highlighted evidence for this answer."
                )
            return

        doc = (
            fitz.open(stream=pdf_bytes, filetype="pdf")
            if pdf_bytes is not None
            else fitz.open(pdf_path)
        )
        try:
            if unscoped_queries:
                inferred_page_queries = _map_queries_to_pages(doc, unscoped_queries)
                for page_number, queries in inferred_page_queries.items():
                    page_queries[page_number].extend(queries)

            if not page_queries:
                st.info(
                    "No matching passage could be highlighted in the source PDF for these chunks."
                )
                return

            page_numbers = sorted(page_queries.keys())[:5]
            for page_number in page_numbers:
                if page_number < 1 or page_number > doc.page_count:
                    continue

                page = doc[page_number - 1]
                deduplicated_queries = list(dict.fromkeys(page_queries[page_number]))
                for query in deduplicated_queries[:5]:
                    rects = _search_rects_with_fallback(page, query)
                    for rect in rects[:3]:
                        annotation = page.add_highlight_annot(rect)
                        annotation.update()

                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                st.image(
                    pixmap.tobytes("png"),
                    caption=f"{filename} - page {page_number}",
                    width="stretch",
                )
        finally:
            doc.close()
