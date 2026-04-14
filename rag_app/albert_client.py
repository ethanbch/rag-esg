from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


@dataclass(frozen=True)
class AlbertApiConfig:
    api_key: str
    base_url: str

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


class AlbertEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, api_config: AlbertApiConfig, model: str) -> None:
        self._api_config = api_config
        self._model = model
        self._embeddings_client = AlbertEmbeddingsClient(
            api_config=api_config,
            model=model,
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self._embeddings_client.embed(list(input))


class AlbertEmbeddingsClient:
    def __init__(self, api_config: AlbertApiConfig, model: str) -> None:
        self._api_config = api_config
        self._model = model

    def embed(self, texts: Sequence[str]) -> Embeddings:
        if not texts:
            return []

        response = requests.post(
            f"{self._api_config.base_url}/embeddings",
            headers=self._api_config.headers(),
            json={"input": list(texts), "model": self._model},
            timeout=60,
        )
        _raise_for_status_with_body(response, context="embedding")

        payload = response.json()
        data = payload.get("data", [])
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding API returned an unexpected number of vectors."
            )

        return embeddings


class AlbertRerankerClient:
    def __init__(self, api_config: AlbertApiConfig, model: str) -> None:
        self._api_config = api_config
        self._model = model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must not be empty")
        if not documents:
            return []

        document_list = list(documents)

        payload_variants = [
            {
                "model": self._model,
                "query": clean_query,
                "documents": document_list,
            },
            {
                "model": self._model,
                "query": clean_query,
                "input": document_list,
            },
        ]
        endpoints = ("/rerank", "/rerankings")
        errors: list[str] = []

        for endpoint in endpoints:
            endpoint_missing = False
            for payload in payload_variants:
                response = requests.post(
                    f"{self._api_config.base_url}{endpoint}",
                    headers=self._api_config.headers(),
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 404:
                    endpoint_missing = True
                    break

                if not response.ok:
                    body = response.text.strip()
                    errors.append(f"{endpoint} ({response.status_code}): {body[:300]}")
                    continue

                return _extract_rerank_scores(response.json(), len(document_list))

            if endpoint_missing:
                continue

        if errors:
            raise RuntimeError(
                "Albert rerank request failed. "
                f"Tried model '{self._model}'. Details: {' | '.join(errors)}"
            )

        raise RuntimeError(
            "No rerank endpoint is available on this API (tried /rerank and /rerankings)."
        )


class AlbertChatClient:
    def __init__(self, api_config: AlbertApiConfig, model: str) -> None:
        self._api_config = api_config
        self._model = model

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        response = requests.post(
            f"{self._api_config.base_url}/chat/completions",
            headers=self._api_config.headers(),
            json={
                "model": self._model,
                "messages": list(messages),
                "temperature": temperature,
            },
            timeout=120,
        )
        _raise_for_status_with_body(response, context="chat")

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Chat API returned no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Chat API returned an empty response.")

        return content


def list_available_models(api_config: AlbertApiConfig) -> list[str]:
    endpoints = ("/models", "/chat/models")
    errors: list[str] = []

    for endpoint in endpoints:
        response = requests.get(
            f"{api_config.base_url}{endpoint}",
            headers=api_config.headers(),
            timeout=30,
        )

        if response.status_code == 404:
            continue

        if not response.ok:
            errors.append(f"{endpoint}: {response.status_code}")
            continue

        payload = response.json()
        models = _extract_model_ids(payload)
        if models:
            return sorted(set(models))

    if errors:
        raise RuntimeError(f"Model listing failed ({'; '.join(errors)}).")

    raise RuntimeError("No models were returned by the API.")


def _extract_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("data", "models", "items", "result"):
            if key in payload:
                return _extract_model_ids(payload[key])

        model_id = payload.get("id")
        if isinstance(model_id, str) and model_id.strip():
            return [model_id.strip()]

        return []

    if isinstance(payload, list):
        model_ids: list[str] = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                model_ids.append(item.strip())
                continue

            if not isinstance(item, dict):
                continue

            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    model_ids.append(value.strip())
                    break

        return model_ids

    return []


def _extract_rerank_scores(payload: Any, expected_count: int) -> list[float]:
    entries: Any = payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items"):
            if key in payload:
                entries = payload[key]
                break

    if not isinstance(entries, list):
        raise RuntimeError("Rerank API returned an unexpected payload format.")

    if all(isinstance(item, (int, float)) for item in entries):
        if len(entries) != expected_count:
            raise RuntimeError(
                "Rerank API returned an unexpected number of scores "
                f"({len(entries)} != {expected_count})."
            )
        return [float(item) for item in entries]

    if not all(isinstance(item, dict) for item in entries):
        raise RuntimeError("Rerank API returned malformed ranking entries.")

    dict_entries: list[dict[str, Any]] = [dict(item) for item in entries]

    if any("index" in entry for entry in dict_entries):
        scores = [float("-inf")] * expected_count
        for fallback_index, entry in enumerate(dict_entries):
            raw_index = entry.get("index", fallback_index)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue

            if not (0 <= index < expected_count):
                continue

            score = _extract_score_value(entry)
            if score is None:
                continue

            scores[index] = score

        return scores

    sequential_scores: list[float] = []
    for entry in dict_entries:
        score = _extract_score_value(entry)
        if score is None:
            raise RuntimeError("Rerank API entries are missing score fields.")
        sequential_scores.append(score)

    if len(sequential_scores) != expected_count:
        raise RuntimeError(
            "Rerank API returned an unexpected number of entries "
            f"({len(sequential_scores)} != {expected_count})."
        )

    return sequential_scores


def _extract_score_value(entry: dict[str, Any]) -> float | None:
    for key in ("relevance_score", "score", "similarity", "relevance", "logit"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _raise_for_status_with_body(response: requests.Response, context: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        raise RuntimeError(f"Albert {context} request failed: {body}") from exc
