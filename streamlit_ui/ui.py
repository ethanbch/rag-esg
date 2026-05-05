from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any

import streamlit as st

from rag_app.prompts import DEFAULT_SYSTEM_PROMPT
from rag_app.service import DEFAULT_CHAT_MODEL, RagRuntimeConfig, RagService
from rag_app.types import RetrievedChunk

from .pdf_evidence import _render_pdf_evidence_for_chunks
from .state import _ingest_uploaded_pdf, _reload_models, _reset_advanced_settings


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
        "What is Airbus's absolute CO2 emissions reduction target for Scope 1 and 2 by 2030 compared to the 2015 baseline?",
        "How many tonnes of CO₂ emissions does Schneider Electric claim to reduce annually by equipping the Kiefer photovoltaic park in Greece?",
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
