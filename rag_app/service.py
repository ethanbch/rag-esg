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

    def _generate_multi_queries(self, question: str, num_queries: int = 3) -> list[str]:
        """Génère des reformulations propres et nettoyées pour le MQR."""
        system_prompt = (
            "Tu es un expert en recherche d'information ESG. "
            "Ta tâche est de reformuler la question de l'utilisateur en générant "
            f"{num_queries} questions alternatives pour maximiser les chances de "
            "trouver des documents pertinents dans une base vectorielle.\n"
            "RÈGLES STRICTES :\n"
            "1. Renvoie UNIQUEMENT les questions, une par ligne.\n"
            "2. N'ajoute AUCUN texte avant ou après.\n"
            "3. N'utilise pas de tirets, de puces ou de numéros au début des lignes."
        )

        try:
            response = self._chat.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question originale: {question}"},
                ],
                temperature=0.4,  # Un peu de créativité pour la diversité sémantique
            )

            # Nettoyage robuste des lignes retournées
            queries = []
            for line in response.split("\n"):
                clean_line = line.strip(" -*1234567890.")
                if (
                    clean_line and len(clean_line) > 10
                ):  # Évite les lignes vides ou trop courtes
                    queries.append(clean_line)

            # On garantit que la question originale est toujours en première position
            return [question] + queries[:num_queries]

        except Exception as e:
            print(
                f"[Avertissement] Échec de la génération MQR, fallback sur la question standard : {e}"
            )
            return [question]

    def retrieve_chunks_mqr(
        self,
        question: str,
        collection_names: list[str],
        n_results_per_collection: int = 3,
        max_chunks: int = 8,
        reranker_backend: str | None = None,
        reranker_candidate_pool: int | None = None,
        min_rerank_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Exécute le Multi-Query Retrieval avec déduplication et reranking global."""

        effective_reranker_candidate_pool = (
            reranker_candidate_pool
            if reranker_candidate_pool is not None
            else self._config.reranker_candidate_pool
        )
        # Sécurité architecturale : Si rien n'est précisé (script d'éval par ex), on fixe une limite physique stricte
        # pour éviter d'inonder le Cross-Encoder
        if effective_reranker_candidate_pool is None:
            effective_reranker_candidate_pool = 60

        effective_min_rerank_score = (
            min_rerank_score
            if min_rerank_score is not None
            else self._config.min_rerank_score
        )

        # 1. Obtenir toutes les requêtes (Originale + Variations)
        queries = self._generate_multi_queries(question)
        print(f"--- MQR activé : Recherche sur {len(queries)} requêtes ---")
        for q in queries:
            print(f"  -> {q}")

        # 2. Phase de Retrieval (Bi-encoder / Recherche Rapide)
        all_retrieved_chunks: list[RetrievedChunk] = []

        for q in queries:
            chunks = self._retriever.retrieve(
                question=q,
                collection_names=collection_names,
                n_results_per_collection=n_results_per_collection,
                max_chunks=15,  # Bon équilibre pour Recall du Bi-encoder au lieu d'inonder la ram
                reranker_backend="none",  # DÉSACTIVÉ : On ne rerank pas encore !
            )
            all_retrieved_chunks.extend(chunks)

        # 3. Déduplication par ID de chunk
        unique_chunks_dict = {chunk.chunk_id: chunk for chunk in all_retrieved_chunks}
        unique_chunks = list(unique_chunks_dict.values())
        print(
            f"--- MQR : {len(all_retrieved_chunks)} chunks bruts -> {len(unique_chunks)} chunks uniques ---"
        )

        if not unique_chunks:
            return []

        if effective_reranker_candidate_pool is not None:
            # Pour éviter de surcharger l'API Reranker, on limite aux meilleurs du bi-encoder
            unique_chunks.sort(
                key=lambda c: c.distance if c.distance is not None else float("inf")
            )
            unique_chunks = unique_chunks[:effective_reranker_candidate_pool]

        # 4. Phase de Reranking (Cross-encoder / Tri Fin)
        effective_reranker = reranker_backend or self._config.reranker_backend

        if effective_reranker in {"api", "cosine"}:
            print("--- MQR : Application du Reranker sur le pool fusionné ---")

            # On évalue tout le pool par rapport à la QUESTION ORIGINALE
            reranked_chunks = self._retriever._rerank_with_api(
                question=question, chunks=unique_chunks
            )

            # Application du seuil minimum si défini
            min_score = (
                effective_min_rerank_score
                if effective_min_rerank_score is not None
                else float("-inf")
            )
            filtered_chunks = [
                c
                for c in reranked_chunks
                if (c.score if c.score is not None else float("-inf")) >= min_score
            ]

            return filtered_chunks[:max_chunks]

        else:
            # Fallback si pas de reranker : on trie par la distance Chroma d'origine
            unique_chunks.sort(
                key=lambda c: c.distance if c.distance is not None else float("inf")
            )
            return unique_chunks[:max_chunks]

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

        # Remplacement de retrieve_chunks par retrieve_chunks_mqr
        chunks = self.retrieve_chunks_mqr(
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
