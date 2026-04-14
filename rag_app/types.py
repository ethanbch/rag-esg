from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    collection_name: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
