from __future__ import annotations

from dataclasses import dataclass

from config import BASE_URL, DB_DIR, EMBEDDING_MODEL, RERANK_BACKEND, RERANK_MODEL

from .albert_client import AlbertApiConfig, AlbertChatClient
from .prompts import build_messages
from .retriever import ChromaRetriever
from .types import RagAnswer, RetrievedChunk

DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class RagRuntimeConfig:
    api_key: str
    model: str = DEFAULT_CHAT_MODEL
    base_url: str = BASE_URL
    db_dir: str = DB_DIR
    embedding_model: str = EMBEDDING_MODEL
    reranker_backend: str = RERANK_BACKEND
    reranker_model: str = RERANK_MODEL
    reranker_candidate_pool: int | None = None
    min_rerank_score: float | None = None


class RagService:
    def __init__(self, config: RagRuntimeConfig) -> None:
        if not config.api_key.strip():
            raise ValueError("api_key must not be empty")

        self._config = config
        api_config = AlbertApiConfig(api_key=config.api_key, base_url=config.base_url)

        self._retriever = ChromaRetriever(
            db_dir=config.db_dir,
            api_config=api_config,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
        )
        self._chat = AlbertChatClient(api_config=api_config, model=config.model)

    def list_collections(self) -> list[str]:
        return self._retriever.list_collections()

    def retrieve_chunks(
        self,
        question: str,
        collection_names: list[str],
        n_results_per_collection: int = 3,
        max_chunks: int = 8,
        reranker_backend: str | None = None,
        reranker_candidate_pool: int | None = None,
        min_rerank_score: float | None = None,
    ) -> list[RetrievedChunk]:
        effective_reranker_backend = reranker_backend or self._config.reranker_backend
        effective_reranker_candidate_pool = (
            reranker_candidate_pool
            if reranker_candidate_pool is not None
            else self._config.reranker_candidate_pool
        )
        effective_min_rerank_score = (
            min_rerank_score
            if min_rerank_score is not None
            else self._config.min_rerank_score
        )

        return self._retriever.retrieve(
            question=question,
            collection_names=collection_names,
            n_results_per_collection=n_results_per_collection,
            max_chunks=max_chunks,
            reranker_backend=effective_reranker_backend,
            reranker_candidate_pool=effective_reranker_candidate_pool,
            min_rerank_score=effective_min_rerank_score,
        )

    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        system_prompt: str | None = None,
    ) -> str:
        messages = build_messages(
            question=question,
            chunks=chunks,
            system_prompt=system_prompt,
        )
        return self._chat.complete(messages=messages, temperature=0.1)

    def ask(
        self,
        question: str,
        collection_names: list[str],
        n_results_per_collection: int = 3,
        max_chunks: int = 8,
        system_prompt: str | None = None,
        reranker_backend: str | None = None,
        reranker_candidate_pool: int | None = None,
        min_rerank_score: float | None = None,
    ) -> RagAnswer:
        chunks = self.retrieve_chunks(
            question=question,
            collection_names=collection_names,
            n_results_per_collection=n_results_per_collection,
            max_chunks=max_chunks,
            reranker_backend=reranker_backend,
            reranker_candidate_pool=reranker_candidate_pool,
            min_rerank_score=min_rerank_score,
        )
        answer = self.generate_answer(
            question=question,
            chunks=chunks,
            system_prompt=system_prompt,
        )
        return RagAnswer(question=question, answer=answer, chunks=chunks)
