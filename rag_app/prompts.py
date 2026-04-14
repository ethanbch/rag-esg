from .types import RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are an ESG RAG assistant. "
    "Answer strictly from the provided context snippets. "
    "If context is missing or uncertain, state it clearly. "
    "Answer in the same language as the user question."
)


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    context = _build_context(chunks)
    resolved_system_prompt = (
        system_prompt.strip()
        if system_prompt is not None and system_prompt.strip()
        else DEFAULT_SYSTEM_PROMPT
    )

    user_prompt = (
        "Context snippets:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Provide a concise answer and cite snippet numbers like [S1], [S2]."
    )

    return [
        {"role": "system", "content": resolved_system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No snippets were retrieved from the knowledge base."

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source = str(chunk.metadata.get("source", chunk.collection_name))
        lines.append(f"[S{idx}] Collection={chunk.collection_name} | Source={source}")
        lines.append(chunk.content)
        lines.append("")

    return "\n".join(lines).strip()
