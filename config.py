import os

from dotenv import load_dotenv

load_dotenv()


def _get_int_env(var_name: str, default: int) -> int:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
DB_DIR = "./chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
MIN_CHUNK_TOKENS = _get_int_env("MIN_CHUNK_TOKENS", 200)
MAX_CHUNK_TOKENS = _get_int_env("MAX_CHUNK_TOKENS", 350)
CHUNK_OVERLAP_TOKENS = _get_int_env("CHUNK_OVERLAP_TOKENS", 80)

CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "token_overlap")
SENTENCE_CHUNK_SIZE = _get_int_env("SENTENCE_CHUNK_SIZE", 8)
SENTENCE_CHUNK_OVERLAP = _get_int_env("SENTENCE_CHUNK_OVERLAP", 2)

RERANK_BACKEND = os.getenv("RERANK_BACKEND", "api")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {ALBERT_API_KEY}"}
