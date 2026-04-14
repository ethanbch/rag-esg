from __future__ import annotations

import math
from collections.abc import Iterable

import chromadb

from .albert_client import (
    AlbertApiConfig,
    AlbertEmbeddingFunction,
    AlbertRerankerClient,
)
from .types import RetrievedChunk


class ChromaRetriever:
    def __init__(
        self,
        db_dir: str,
        api_config: AlbertApiConfig,
        embedding_model: str,
        reranker_model: str,
    ) -> None:
        self._client = chromadb.PersistentClient(path=db_dir)
        self._api_config = api_config
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model

    def list_collections(self) -> list[str]:
        return sorted(collection.name for collection in self._client.list_collections())

    def retrieve(
        self,
        question: str,
        collection_names: Iterable[str],
        n_results_per_collection: int = 3,
        max_chunks: int = 8,
        reranker_backend: str = "cosine",
        reranker_candidate_pool: int | None = None,
        min_rerank_score: float | None = None,
    ) -> list[RetrievedChunk]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question must not be empty")
        if n_results_per_collection <= 0:
            raise ValueError("n_results_per_collection must be > 0")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")
        if reranker_candidate_pool is not None and reranker_candidate_pool <= 0:
            raise ValueError("reranker_candidate_pool must be > 0")

        embedding_function = AlbertEmbeddingFunction(
            api_config=self._api_config,
            model=self._embedding_model,
        )

        chunks: list[RetrievedChunk] = []

        for collection_name in collection_names:
            collection = self._client.get_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
            results = collection.query(
                query_texts=[clean_question],
                n_results=n_results_per_collection,
                include=["documents", "metadatas", "distances"],
            )

            ids = (results.get("ids") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]

            for idx, chunk_id in enumerate(ids):
                content = documents[idx] if idx < len(documents) else ""

                metadata = (
                    metadatas[idx]
                    if idx < len(metadatas) and isinstance(metadatas[idx], dict)
                    else {}
                )
                distance = distances[idx] if idx < len(distances) else None

                chunks.append(
                    RetrievedChunk(
                        collection_name=collection_name,
                        chunk_id=chunk_id,
                        content=content,
                        metadata=metadata,
                        distance=distance,
                    )
                )

        if not chunks:
            return []

        normalized_backend = reranker_backend.strip().lower()
        if normalized_backend == "none":
            return self._sort_by_distance(chunks)[:max_chunks]
        if normalized_backend in {"api", "cosine"}:
            chunks_for_rerank = self._sort_by_distance(chunks)
            if reranker_candidate_pool is not None:
                chunks_for_rerank = chunks_for_rerank[:reranker_candidate_pool]

            reranked_chunks = self._rerank_with_api(
                question=clean_question, chunks=chunks_for_rerank
            )
            effective_min_rerank_score = (
                float(min_rerank_score)
                if min_rerank_score is not None
                else float("-inf")
            )
            filtered_chunks = [
                chunk
                for chunk in reranked_chunks
                if (chunk.score if chunk.score is not None else float("-inf"))
                >= effective_min_rerank_score
            ]
            return filtered_chunks[:max_chunks]

        raise ValueError("Unsupported reranker backend. Use 'api' or 'none'.")

    @staticmethod
    def _sort_by_distance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.distance if chunk.distance is not None else math.inf
            ),
        )

    def _rerank_with_api(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        reranker_client = AlbertRerankerClient(
            api_config=self._api_config,
            model=self._reranker_model,
        )
        scores = reranker_client.rerank(
            query=question,
            documents=[chunk.content for chunk in chunks],
        )

        reranked_chunks: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores):
            reranked_chunks.append(
                RetrievedChunk(
                    collection_name=chunk.collection_name,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    distance=chunk.distance,
                    score=score,
                )
            )

        reranked_chunks.sort(
            key=lambda chunk: chunk.score if chunk.score is not None else float("-inf"),
            reverse=True,
        )
        return reranked_chunks


def list_local_collection_names(db_dir: str) -> list[str]:
    client = chromadb.PersistentClient(path=db_dir)
    return sorted(collection.name for collection in client.list_collections())
