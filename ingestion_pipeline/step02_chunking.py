import os
import re

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNKING_STRATEGY,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    SENTENCE_CHUNK_OVERLAP,
    SENTENCE_CHUNK_SIZE,
)

TOKEN_PATTERN = re.compile(r"\S+")
FALLBACK_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:  # pragma: no cover
    nltk = None
    sent_tokenize = None


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def _sentence_tokenize_with_nltk(text: str) -> list[str] | None:
    if nltk is None or sent_tokenize is None:
        return None

    try:
        return [
            sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()
        ]
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

        try:
            return [
                sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()
            ]
        except Exception:
            return None
    except Exception:
        return None


def _sentence_tokenize(text: str) -> list[str]:
    clean_text = text.strip()
    if not clean_text:
        return []

    nltk_sentences = _sentence_tokenize_with_nltk(clean_text)
    if nltk_sentences:
        return nltk_sentences

    sentences = [
        sentence.strip()
        for sentence in FALLBACK_SENTENCE_PATTERN.split(clean_text)
        if sentence.strip()
    ]
    return sentences if sentences else [clean_text]


def _chunk_tokens(
    tokens: list[str], min_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[list[str]]:
    if not tokens:
        return []

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= min_tokens:
        raise ValueError("overlap_tokens must be strictly lower than min_tokens")

    chunks: list[list[str]] = []
    index = 0
    total = len(tokens)
    step = max_tokens - overlap_tokens

    while index < total:
        remaining = total - index
        if remaining <= max_tokens:
            chunks.append(tokens[index:])
            break

        chunk_end = index + max_tokens
        chunks.append(tokens[index:chunk_end])
        index += step

    # Merge too-short trailing chunk with previous chunk to keep chunks >= min_tokens when possible.
    if len(chunks) >= 2 and len(chunks[-1]) < min_tokens:
        previous_chunk = chunks[-2]
        short_chunk = chunks[-1]
        merged = previous_chunk + short_chunk
        if len(merged) <= max_tokens:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def _chunk_sentences(
    sentences: list[str],
    target_sentences: int,
    overlap_sentences: int,
) -> list[list[str]]:
    if not sentences:
        return []

    if target_sentences <= 0:
        raise ValueError("target_sentences must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")
    if overlap_sentences >= target_sentences:
        raise ValueError(
            "overlap_sentences must be strictly lower than target_sentences"
        )

    chunks: list[list[str]] = []
    index = 0
    total = len(sentences)
    step = target_sentences - overlap_sentences

    while index < total:
        remaining = total - index
        if remaining <= target_sentences:
            chunks.append(sentences[index:])
            break

        chunk_end = index + target_sentences
        chunks.append(sentences[index:chunk_end])
        index += step

    return chunks


def chunk_text(
    text: str,
    source_name: str,
    doc_id: str,
    strategy: str = CHUNKING_STRATEGY,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    sentence_chunk_size: int = SENTENCE_CHUNK_SIZE,
    sentence_chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
) -> list[dict]:
    if not text.strip():
        return []

    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "token_overlap":
        tokens = _tokenize(text)
        if not tokens:
            return []

        token_chunks = _chunk_tokens(tokens, min_tokens, max_tokens, overlap_tokens)

        return [
            {
                "id": f"{doc_id}_chunk_{i}",
                "content": _detokenize(token_chunk),
                "metadata": {
                    "source": source_name,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "token_count": len(token_chunk),
                    "sentence_count": len(_sentence_tokenize(_detokenize(token_chunk))),
                    "chunking_strategy": normalized_strategy,
                },
            }
            for i, token_chunk in enumerate(token_chunks)
        ]

    if normalized_strategy == "sentence_overlap":
        sentences = _sentence_tokenize(text)
        if not sentences:
            return []

        sentence_chunks = _chunk_sentences(
            sentences=sentences,
            target_sentences=sentence_chunk_size,
            overlap_sentences=sentence_chunk_overlap,
        )

        chunks: list[dict] = []
        for i, sentence_chunk in enumerate(sentence_chunks):
            content = " ".join(sentence_chunk)
            token_count = len(_tokenize(content))
            chunks.append(
                {
                    "id": f"{doc_id}_chunk_{i}",
                    "content": content,
                    "metadata": {
                        "source": source_name,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "token_count": token_count,
                        "sentence_count": len(sentence_chunk),
                        "chunking_strategy": normalized_strategy,
                    },
                }
            )

        return chunks

    raise ValueError(
        "Unsupported chunking strategy. Use 'token_overlap' or 'sentence_overlap'."
    )


def chunk_pdf_text(
    pdf_path: str,
    text: str,
    strategy: str = CHUNKING_STRATEGY,
) -> list[dict]:
    source_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(source_name)[0]
    chunks = chunk_text(
        text=text,
        source_name=source_name,
        doc_id=doc_id,
        strategy=strategy,
    )

    if chunks:
        sizes = [chunk["metadata"]["token_count"] for chunk in chunks]
        selected_strategy = chunks[0]["metadata"].get("chunking_strategy", strategy)
        print(
            "    Chunked to "
            f"{len(chunks)} chunks (strategy={selected_strategy}, min={min(sizes)} tokens, max={max(sizes)} tokens)."
        )

    return chunks
