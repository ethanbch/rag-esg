import chromadb
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

from config import BASE_URL, DB_DIR, EMBEDDING_MODEL, get_headers


class AlbertEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = EMBEDDING_MODEL, api_key: str | None = None):
        self._model = model
        self._api_key = api_key.strip() if isinstance(api_key, str) else None

    def __call__(self, input: Documents) -> Embeddings:
        headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key
            else get_headers()
        )
        res = requests.post(
            f"{BASE_URL}/embeddings",
            headers=headers,
            json={"input": input, "model": self._model},
        )
        res.raise_for_status()
        data = res.json().get("data", [])
        return [item["embedding"] for item in data]


def index_chunks(
    collection_name: str,
    chunks: list[dict],
    embedding_model: str = EMBEDDING_MODEL,
    replace_collection: bool = True,
    api_key: str | None = None,
) -> None:
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    if replace_collection:
        # Replace collection to avoid embedding-function conflicts across runs.
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass

    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=AlbertEmbeddingFunction(
            model=embedding_model,
            api_key=api_key,
        ),
    )

    ids = [
        f"{chunk['metadata']['doc_id']}_{index}" for index, chunk in enumerate(chunks)
    ]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    batch_size = 50
    for i in tqdm(range(0, len(ids), batch_size), desc="Embedding & Storing chunks"):
        chroma_collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
