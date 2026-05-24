from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from config import (
    ALBERT_API_KEY,
    BASE_URL,
    DB_DIR,
    EMBEDDING_MODEL,
    RERANK_BACKEND,
    RERANK_MODEL,
)
from ingestion_pipeline.main import process_pdf
from ingestion_pipeline.step01_parsing import extract_text_by_page
from rag_app.albert_client import AlbertApiConfig, list_available_models
from rag_app.prompts import DEFAULT_SYSTEM_PROMPT
from rag_app.retriever import list_local_collection_names
from rag_app.service import DEFAULT_CHAT_MODEL


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
        "n_results_per_collection": 10,
        "max_chunks": 50,
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
    st.session_state.n_results_per_collection = 10
    st.session_state.max_chunks = 50
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


def _build_highlight_queries(text: str, question: str = "") -> list[str]:
    """Retourne plusieurs phrases candidates à surligner depuis un chunk."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    # Découpe en phrases
    sentences = re.split(r"(?<=[.!?])\s+|(?<=;)\s+|(?<=\n)•\s*", normalized)
    question_words = set(re.findall(r"\w{4,}", question.lower()))

    scored = []
    for s in sentences:
        s = s.strip().lstrip("•·-– ")
        words = s.split()
        if len(words) < 3 or len(words) > 25:
            continue
        # Score: mots en commun avec la question + bonus si chiffre
        overlap = sum(
            1 for w in re.findall(r"\w{4,}", s.lower()) if w in question_words
        )
        has_number = bool(re.search(r"\d", s))
        score = overlap * 2 + (1 if has_number else 0)
        scored.append((score, len(words), " ".join(words[:16])))

    scored.sort(key=lambda x: (-x[0], x[1]))

    seen = set()
    out = []
    for _, _, phrase in scored:
        if phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
        if len(out) >= 5:
            break

    # Fallback
    return out or [" ".join(normalized.split()[:12])]


def _ingest_uploaded_pdf(uploaded_file: Any) -> tuple[str, int, int]:
    filename = str(getattr(uploaded_file, "name", "uploaded.pdf"))
    pdf_bytes = bytes(uploaded_file.getvalue())
    if not pdf_bytes:
        raise RuntimeError("Uploaded file is empty.")
    collection_name = _build_upload_collection_name(filename, pdf_bytes)

    page_count = 0
    chunk_count = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        tmp_path.write_bytes(pdf_bytes)

        pages = extract_text_by_page(str(tmp_path))
        page_count = len(pages)

        chunk_count = (
            process_pdf(
                str(tmp_path),
                collection_name,
                api_key=str(st.session_state.api_key),
                embedding_model=str(st.session_state.embedding_model),
            )
            or 0
        )

    if chunk_count <= 0:
        raise RuntimeError("No chunks generated from uploaded PDF.")

    st.session_state.available_collections = _reload_collections()
    st.session_state.selected_collections = [collection_name]

    uploaded_documents = dict(st.session_state.uploaded_documents)
    uploaded_documents[collection_name] = {
        "filename": filename,
        "pdf_bytes": pdf_bytes,
    }
    st.session_state.uploaded_documents = uploaded_documents

    return collection_name, page_count, chunk_count


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
