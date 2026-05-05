from .types import RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert ESG assistant. "
    "Answer strictly using the provided context snippets. "
    "If the information is missing or uncertain, state it clearly. "
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

    # Modification de l'instruction pour exiger une citation de la source réelle
    user_prompt = (
        "Context snippets:\n"
        f"{context}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Provide a concise answer. Always justify your statements by citing the "
        "specific SOURCE name in parentheses (e.g., '(Source: 2024_ESG_Report)' "
        "or 'According to [Source Name]...'). Do not use generic snippet numbers."
    )

    return [
        {"role": "system", "content": resolved_system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No snippets were retrieved from the knowledge base."

    lines: list[str] = []

    # Plus besoin de la fonction enumerate(..., start=1)
    for chunk in chunks:
        source = str(chunk.metadata.get("source", chunk.collection_name))

        # On utilise un délimiteur clair pour indiquer le nom de la source au LLM
        lines.append(f"--- SOURCE: {source} ---")
        lines.append(chunk.content)
        lines.append("")

    return "\n".join(lines).strip()
