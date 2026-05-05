from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st

from config import (
    ALBERT_API_KEY,
    BASE_URL,
    CHUNKING_STRATEGY,
    DB_DIR,
    EMBEDDING_MODEL,
    RERANK_BACKEND,
    RERANK_MODEL,
)
from ingestion_pipeline.step01_parsing import extract_text_by_page
from ingestion_pipeline.step02_chunking import chunk_text
from ingestion_pipeline.step03_indexing import index_chunks
from rag_app.albert_client import AlbertApiConfig, list_available_models
from rag_app.prompts import DEFAULT_SYSTEM_PROMPT
from rag_app.retriever import list_local_collection_names
from rag_app.service import DEFAULT_CHAT_MODEL, RagRuntimeConfig, RagService
from rag_app.types import RetrievedChunk

st.set_page_config(
    page_title="ESG RAG Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "api_key": ALBERT_API_KEY or "",
        "model_name": DEFAULT_CHAT_MODEL,
        "available_collections": [],
        "selected_collections": [],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "available_models": [],
        "models_api_key": "",
        "models_error": "",
        "chat_messages": [],
        "embedding_model": EMBEDDING_MODEL,
        "reranker_backend": RERANK_BACKEND,
        "reranker_model": RERANK_MODEL,
        "n_results_per_collection": 5,
        "max_chunks": 12,
        "reranker_candidate_pool": 60,
        "min_rerank_score": 0.1,
        "queued_prompt": None,
        "uploaded_documents": {},
        "uploader_key": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, list):
                st.session_state[key] = list(value)
            elif isinstance(value, dict):
                st.session_state[key] = dict(value)
            else:
                st.session_state[key] = value


def _reset_advanced_settings() -> None:
    default_reranker_backend = str(RERANK_BACKEND).strip().lower()
    if default_reranker_backend == "cosine":
        default_reranker_backend = "api"
    if default_reranker_backend not in {"api", "none"}:
        default_reranker_backend = "api"

    st.session_state.embedding_model = EMBEDDING_MODEL
    st.session_state.reranker_backend = default_reranker_backend
    st.session_state.reranker_model = RERANK_MODEL
    st.session_state.n_results_per_collection = 5
    st.session_state.max_chunks = 12
    st.session_state.reranker_candidate_pool = 60
    st.session_state.min_rerank_score = 0.1
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT


def _reload_collections() -> list[str]:
    try:
        return list_local_collection_names(DB_DIR)
    except Exception as exc:
        st.error(f"Could not read ChromaDB collections: {exc}")
        return []


def _reload_models(api_key: str) -> list[str]:
    clean_key = api_key.strip()
    if not clean_key:
        st.session_state.models_error = "Add API key to load models from API."
        return [DEFAULT_CHAT_MODEL]

    try:
        api_config = AlbertApiConfig(api_key=clean_key, base_url=BASE_URL)
        models = list_available_models(api_config)
    except Exception as exc:
        st.session_state.models_error = f"Could not load models from API: {exc}"
        return [DEFAULT_CHAT_MODEL]

    st.session_state.models_error = ""
    return list(dict.fromkeys([DEFAULT_CHAT_MODEL, *models]))


def _sanitize_collection_name(raw_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    if not cleaned:
        cleaned = "uploaded_pdf"
    if not cleaned[0].isalnum():
        cleaned = f"doc_{cleaned}"

    return cleaned[:120]


def _build_upload_collection_name(filename: str, pdf_bytes: bytes) -> str:
    stem = Path(filename).stem
    base = _sanitize_collection_name(stem)
    digest = hashlib.sha1(pdf_bytes).hexdigest()[:8]
    return _sanitize_collection_name(f"{base}_{digest}")


def _build_highlight_query(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    words = normalized.split()
    if not words:
        return ""
    return " ".join(words[:24])


def _extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> list[dict[str, str | int]]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        return extract_text_by_page(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _ingest_uploaded_pdf(uploaded_file: Any) -> tuple[str, int, int]:
    filename = str(getattr(uploaded_file, "name", "uploaded.pdf"))
    pdf_bytes = bytes(uploaded_file.getvalue())
    if not pdf_bytes:
        raise RuntimeError("Uploaded file is empty.")

    pages = _extract_pages_from_pdf_bytes(pdf_bytes)
    if not pages:
        raise RuntimeError("No extractable text found in uploaded PDF.")

    collection_name = _build_upload_collection_name(filename, pdf_bytes)
    doc_id_root = _sanitize_collection_name(Path(filename).stem)

    all_chunks: list[dict[str, Any]] = []
    for page_payload in pages:
        page_number_raw = page_payload.get("page_number")
        page_text_raw = page_payload.get("text")
        if not isinstance(page_text_raw, str) or not page_text_raw.strip():
            continue

        try:
            page_number = int(page_number_raw)
        except (TypeError, ValueError):
            continue

        page_chunks = chunk_text(
            text=page_text_raw,
            source_name=filename,
            doc_id=f"{doc_id_root}_p{page_number}",
            strategy=CHUNKING_STRATEGY,
        )

        for chunk in page_chunks:
            metadata = (
                chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            )
            metadata["page_number"] = page_number
            metadata["source_file"] = filename
            metadata["highlight_query"] = _build_highlight_query(
                str(chunk.get("content", ""))
            )
            chunk["metadata"] = metadata
            all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError("No chunks generated from uploaded PDF.")

    index_chunks(
        collection_name=collection_name,
        chunks=all_chunks,
        embedding_model=str(st.session_state.embedding_model),
        replace_collection=True,
        api_key=str(st.session_state.api_key),
    )

    st.session_state.available_collections = _reload_collections()
    st.session_state.selected_collections = [collection_name]

    uploaded_documents = dict(st.session_state.uploaded_documents)
    uploaded_documents[collection_name] = {
        "filename": filename,
        "pdf_bytes": pdf_bytes,
    }
    st.session_state.uploaded_documents = uploaded_documents

    return collection_name, len(pages), len(all_chunks)


def _select_primary_pdf_filename(chunks: list[dict[str, Any]]) -> str | None:
    filename_counts: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        metadata = (
            chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        )
        for key in ("source_file", "source"):
            raw_value = str(metadata.get(key, "")).strip()
            if not raw_value:
                continue

            basename = os.path.basename(raw_value)
            if basename.lower().endswith(".pdf"):
                filename_counts[basename] += 1
                break

    if not filename_counts:
        return None

    return max(filename_counts.items(), key=lambda item: item[1])[0]


def _resolve_local_pdf_path(pdf_filename: str) -> Path | None:
    clean_name = os.path.basename(pdf_filename.strip())
    if not clean_name:
        return None

    direct_candidates = [Path(clean_name), Path("downloads") / clean_name]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    downloads_dir = Path("downloads")
    if downloads_dir.exists() and downloads_dir.is_dir():
        lowered_name = clean_name.lower()
        for candidate in downloads_dir.glob("*.pdf"):
            if candidate.name.lower() == lowered_name:
                return candidate

    return None


def _resolve_pdf_source_for_chunks(
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    uploaded_documents = st.session_state.get("uploaded_documents", {})
    if isinstance(uploaded_documents, dict) and uploaded_documents:
        uploaded_collection_counts: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            collection_name = str(chunk.get("collection_name", ""))
            if collection_name in uploaded_documents:
                uploaded_collection_counts[collection_name] += 1

        if uploaded_collection_counts:
            selected_collection = max(
                uploaded_collection_counts.items(), key=lambda item: item[1]
            )[0]
            document_payload = uploaded_documents.get(selected_collection, {})
            if isinstance(document_payload, dict):
                pdf_bytes = document_payload.get("pdf_bytes")
                filename = str(document_payload.get("filename", "uploaded.pdf"))
                if isinstance(pdf_bytes, (bytes, bytearray)) and pdf_bytes:
                    return {
                        "filename": filename,
                        "pdf_bytes": bytes(pdf_bytes),
                    }

    pdf_filename = _select_primary_pdf_filename(chunks)
    if not pdf_filename:
        return None

    local_path = _resolve_local_pdf_path(pdf_filename)
    if local_path is None:
        return None

    return {
        "filename": pdf_filename,
        "pdf_path": str(local_path),
    }


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


def _chunk_to_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "collection_name": chunk.collection_name,
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": chunk.metadata,
        "distance": chunk.distance,
        "score": chunk.score,
    }


def _render_header() -> None:
    selected_collection_count = len(st.session_state.selected_collections)
    available_collection_count = len(st.session_state.available_collections)
    reranker_enabled = str(st.session_state.reranker_backend).strip().lower() != "none"
    model_name = str(st.session_state.model_name or DEFAULT_CHAT_MODEL)
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    st.title("ESG Assistant")
    st.caption("Ask questions on your corpus and inspect grounded evidence.")

    status_col_1, status_col_2, status_col_3, status_col_4 = st.columns(4)
    with status_col_1:
        st.metric("Scope", f"{selected_collection_count} selected")
    with status_col_2:
        st.metric(
            "Top K",
            f"{int(st.session_state.n_results_per_collection)} / collection",
        )
    with status_col_3:
        st.metric("Final Top K", f"{int(st.session_state.max_chunks)} chunks")
    with status_col_4:
        st.metric("Reranker", "On" if reranker_enabled else "Off")

    preview_names = st.session_state.selected_collections[:3]
    if preview_names:
        preview_label = ", ".join(preview_names)
        if selected_collection_count > 3:
            preview_label += f" +{selected_collection_count - 3}"
    else:
        preview_label = "none"

    st.caption(
        f"Model: {model_short} | Collections: {preview_label} | Available: {available_collection_count}"
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Session")
        if st.button("New chat", width="stretch", type="primary"):
            st.session_state.chat_messages = []
            st.rerun()

        st.divider()
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader(
            "Drop a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            key=f"pdf_uploader_{st.session_state.uploader_key}",
        )

        if uploaded_file is not None:
            st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "Ingest & index uploaded PDF",
            width="stretch",
            disabled=uploaded_file is None,
        ):
            if not st.session_state.api_key.strip():
                st.error("Please provide an API key before indexing a PDF.")
            elif uploaded_file is None:
                st.warning("Please select a PDF first.")
            else:
                try:
                    with st.spinner("Parsing, chunking, embedding and indexing PDF..."):
                        collection_name, page_count, chunk_count = _ingest_uploaded_pdf(
                            uploaded_file
                        )
                except Exception as exc:
                    st.error(f"PDF ingestion failed: {exc}")
                else:
                    st.success(
                        f"Indexed {uploaded_file.name} into '{collection_name}' "
                        f"({page_count} pages, {chunk_count} chunks)."
                    )
                    st.session_state.uploader_key = (
                        int(st.session_state.uploader_key) + 1
                    )
                    st.rerun()

        st.divider()
        st.markdown("### Core config")

        st.session_state.api_key = st.text_input(
            "API key",
            value=st.session_state.api_key,
            type="password",
            help="Albert API key used for embeddings and chat/completions.",
        )

        if st.session_state.models_api_key != st.session_state.api_key:
            st.session_state.available_models = _reload_models(st.session_state.api_key)
            st.session_state.models_api_key = st.session_state.api_key

        model_options = st.session_state.available_models or [DEFAULT_CHAT_MODEL]
        if st.session_state.model_name not in model_options:
            st.session_state.model_name = model_options[0]

        st.session_state.model_name = st.selectbox(
            "LLM model",
            options=model_options,
            index=model_options.index(st.session_state.model_name),
            help="Models are loaded dynamically from API.",
        )

        available_collections = st.session_state.available_collections
        if available_collections:
            valid_default = [
                name
                for name in st.session_state.selected_collections
                if name in available_collections
            ]
            default_selection = valid_default or available_collections

            st.session_state.selected_collections = st.multiselect(
                "Documents in the knowledge base",
                options=available_collections,
                default=default_selection,
                help="Select one or more collections to query.",
            )
        else:
            st.warning("No collections found in local ChromaDB.")
            st.session_state.selected_collections = []

        with st.expander("Advanced controls", expanded=False):
            st.session_state.embedding_model = st.text_input(
                "Embedding model",
                value=st.session_state.embedding_model,
                help="Fixed by configuration to avoid runtime mismatches.",
                disabled=True,
            )

            normalized_backend = str(st.session_state.reranker_backend).strip().lower()
            if normalized_backend == "cosine":
                normalized_backend = "api"

            reranker_enabled = st.checkbox(
                "Enable reranker",
                value=normalized_backend != "none",
                help="Use API reranking on top of vector retrieval.",
            )
            st.session_state.reranker_backend = "api" if reranker_enabled else "none"

            st.session_state.reranker_model = st.text_input(
                "Reranker model",
                value=st.session_state.reranker_model,
                help="Fixed by configuration to avoid runtime mismatches.",
                disabled=True,
            )

            st.session_state.n_results_per_collection = st.slider(
                "Top K",
                min_value=1,
                max_value=10,
                value=int(st.session_state.n_results_per_collection),
                help="Number of chunks retrieved per collection before reranking.",
            )
            st.session_state.max_chunks = st.slider(
                "Final Top K",
                min_value=1,
                max_value=30,
                value=int(st.session_state.max_chunks),
                help="Number of chunks kept for final context after reranking/sorting.",
            )

            current_pool_value = int(st.session_state.reranker_candidate_pool)
            min_pool_value = int(st.session_state.max_chunks)
            if current_pool_value < min_pool_value:
                current_pool_value = min_pool_value

            st.session_state.reranker_candidate_pool = st.slider(
                "Reranker candidate pool",
                min_value=min_pool_value,
                max_value=120,
                value=current_pool_value,
                help=(
                    "Maximum number of vector candidates passed to the reranker "
                    "before the final Top K cutoff."
                ),
                disabled=not reranker_enabled,
            )

            current_min_score = float(st.session_state.min_rerank_score)
            current_min_score = min(1.0, max(0.0, current_min_score))
            st.session_state.min_rerank_score = st.slider(
                "Min rerank score",
                min_value=0.0,
                max_value=1.0,
                value=current_min_score,
                step=0.01,
                help="Chunks with a lower rerank score are discarded.",
                disabled=not reranker_enabled,
            )

            if st.button("Reset settings", width="stretch"):
                _reset_advanced_settings()
                st.rerun()

            st.session_state.system_prompt = st.text_area(
                "System prompt",
                value=st.session_state.system_prompt,
                height=180,
                help="System message sent to the generation model.",
            )

            if st.button("Reset prompt", width="stretch"):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.rerun()

        if st.session_state.models_error:
            st.caption(st.session_state.models_error)


def _render_answer_meta(message: dict[str, Any]) -> None:
    model_name = str(message.get("model", ""))
    reranker_backend = str(message.get("reranker", "")).strip().lower()
    reranker_status = ""
    if reranker_backend:
        reranker_status = "Off" if reranker_backend == "none" else "On"
    reranker_model = str(message.get("reranker_model", ""))
    reranker_pool = message.get("reranker_pool")
    min_rerank_score = message.get("min_rerank_score")
    total_ms = message.get("total_ms")
    chunk_count = message.get("chunk_count")

    parts: list[str] = []
    if model_name:
        parts.append(f"Model: {model_name}")
    if reranker_status:
        parts.append(f"Reranker: {reranker_status}")
    if reranker_model:
        parts.append(f"Reranker model: {reranker_model}")
    if isinstance(reranker_pool, int) and reranker_status == "On":
        parts.append(f"Reranker pool: {reranker_pool}")
    if reranker_status == "On" and isinstance(min_rerank_score, (int, float)):
        parts.append(f"Min score: {float(min_rerank_score):.4f}")
    if isinstance(chunk_count, int):
        parts.append(f"Chunks: {chunk_count}")
    if isinstance(total_ms, (int, float)):
        parts.append(f"Latency: {total_ms:.0f} ms")

    if parts:
        st.caption(" | ".join(parts))


def _render_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    with st.expander(f"Evidence chunks ({len(chunks)})", expanded=False):
        for index, chunk in enumerate(chunks, start=1):
            metadata = (
                chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            )
            source = str(metadata.get("source", "unknown"))
            collection_name = str(chunk.get("collection_name", "unknown"))
            page_number_raw = metadata.get("page_number")
            try:
                page_number = int(page_number_raw)
            except (TypeError, ValueError):
                page_number = None
            distance = chunk.get("distance")
            score = chunk.get("score")

            st.markdown(f"#### Chunk {index}")
            caption = f"Collection: {collection_name} | Source: {source}"
            if page_number is not None:
                caption += f" | Page: {page_number}"
            st.caption(caption)

            m1, m2 = st.columns(2)
            with m1:
                if isinstance(score, (int, float)):
                    st.metric("Rerank score", f"{score:.4f}")
                else:
                    st.metric("Rerank score", "n/a")
            with m2:
                if isinstance(distance, (int, float)):
                    st.metric("Vector distance", f"{distance:.4f}")
                else:
                    st.metric("Vector distance", "n/a")

            st.write(str(chunk.get("content", "")))
            if metadata:
                with st.expander("Metadata", expanded=False):
                    st.json(metadata)

            if index < len(chunks):
                st.divider()


def _render_empty_state() -> None:
    st.caption("Pick a prompt below or ask your own ESG question.")

    suggestions = [
        "What are the 4 SDG axes in Sustainab'ALL?",
        "Summarize the 2030 Scope 1+2 trajectory and targets.",
        "What does the report say about taxonomy capex alignment?",
    ]
    cols = st.columns(len(suggestions))
    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"suggestion_{idx}", width="stretch"):
                st.session_state.queued_prompt = suggestion
                st.rerun()


def _render_history() -> None:
    for index, message in enumerate(st.session_state.chat_messages):
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                _render_answer_meta(message)
                chunks = message.get("chunks")
                if isinstance(chunks, list):
                    _render_chunks(chunks)
                    message_id = str(message.get("message_id", f"history_{index}"))
                    _render_pdf_evidence_for_chunks(
                        chunks,
                        evidence_key=message_id,
                        auto_render=False,
                    )


def _generate_answer(clean_question: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": clean_question})

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        if not st.session_state.api_key.strip():
            error_message = "Please provide an API key."
            st.error(error_message)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": error_message}
            )
            return

        if not st.session_state.selected_collections:
            error_message = "Please select at least one ChromaDB collection."
            st.error(error_message)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": error_message}
            )
            return

        runtime_config = RagRuntimeConfig(
            api_key=st.session_state.api_key,
            model=st.session_state.model_name,
            embedding_model=st.session_state.embedding_model,
            reranker_backend=st.session_state.reranker_backend,
            reranker_model=st.session_state.reranker_model,
            reranker_candidate_pool=int(st.session_state.reranker_candidate_pool),
            min_rerank_score=float(st.session_state.min_rerank_score),
        )

        try:
            rag_service = RagService(runtime_config)

            total_start = perf_counter()
            with st.status("RAG pipeline running", expanded=True) as status:
                st.write("1/4 Receive question")

                st.write("2/4 Retrieve chunks")
                retrieval_start = perf_counter()
                chunks = rag_service.retrieve_chunks(
                    question=clean_question,
                    collection_names=st.session_state.selected_collections,
                    n_results_per_collection=int(
                        st.session_state.n_results_per_collection
                    ),
                    max_chunks=int(st.session_state.max_chunks),
                    reranker_backend=st.session_state.reranker_backend,
                    reranker_candidate_pool=int(
                        st.session_state.reranker_candidate_pool
                    ),
                    min_rerank_score=float(st.session_state.min_rerank_score),
                )
                retrieval_ms = (perf_counter() - retrieval_start) * 1000
                st.write(f"Retrieved {len(chunks)} chunk(s) in {retrieval_ms:.0f} ms")

                st.write("3/4 Generate answer")
                generation_start = perf_counter()
                answer = rag_service.generate_answer(
                    question=clean_question,
                    chunks=chunks,
                    system_prompt=st.session_state.system_prompt,
                )
                generation_ms = (perf_counter() - generation_start) * 1000
                st.write(f"Answer generated in {generation_ms:.0f} ms")

                total_ms = (perf_counter() - total_start) * 1000
                st.write("4/4 Render response")
                status.update(
                    label=f"RAG pipeline complete ({total_ms:.0f} ms)",
                    state="complete",
                )
        except Exception as exc:
            error_message = f"RAG pipeline failed: {exc}"
            st.error(error_message)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": error_message}
            )
            return

        chunk_payload = [_chunk_to_payload(chunk) for chunk in chunks]
        message_id = uuid.uuid4().hex
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "model": st.session_state.model_name,
            "reranker": st.session_state.reranker_backend,
            "reranker_model": st.session_state.reranker_model,
            "reranker_pool": int(st.session_state.reranker_candidate_pool),
            "min_rerank_score": float(st.session_state.min_rerank_score),
            "chunk_count": len(chunk_payload),
            "total_ms": total_ms,
            "chunks": chunk_payload,
            "message_id": message_id,
        }

        st.markdown(answer)
        _render_answer_meta(assistant_message)
        _render_chunks(chunk_payload)
        _render_pdf_evidence_for_chunks(
            chunk_payload,
            evidence_key=message_id,
            auto_render=False,
        )
        st.session_state.chat_messages.append(assistant_message)


def main() -> None:
    _initialize_state()

    if not st.session_state.available_collections:
        st.session_state.available_collections = _reload_collections()

    _render_sidebar()
    _render_header()
    _render_history()

    if not st.session_state.chat_messages:
        _render_empty_state()

    queued_prompt = st.session_state.pop("queued_prompt", None)
    chat_prompt = st.chat_input("Ask your ESG question")
    incoming_prompt = queued_prompt if isinstance(queued_prompt, str) else chat_prompt

    if incoming_prompt is None:
        return

    clean_question = incoming_prompt.strip()
    if not clean_question:
        st.warning("Please enter a non-empty question.")
        return

    _generate_answer(clean_question)


if __name__ == "__main__":
    main()
