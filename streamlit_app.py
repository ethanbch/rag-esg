from __future__ import annotations

import streamlit as st

from streamlit_ui.state import _initialize_state, _reload_collections
from streamlit_ui.ui import (
    _generate_answer,
    _render_empty_state,
    _render_header,
    _render_history,
    _render_sidebar,
)

st.set_page_config(
    page_title="ESG RAG Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    _initialize_state()

    if not st.session_state.available_collections:
        st.session_state.available_collections = _reload_collections()

    _render_sidebar()
    _render_header()
    _render_history()

    if not st.session_state.chat_messages:
        _render_empty_state()

    queued_prompt = st.session_state.pop("queued_prompt", None)
    chat_prompt = st.chat_input("Ask your ESG question")
    incoming_prompt = queued_prompt if isinstance(queued_prompt, str) else chat_prompt

    if incoming_prompt is None:
        return

    clean_question = incoming_prompt.strip()
    if not clean_question:
        st.warning("Please enter a non-empty question.")
        return

    _generate_answer(clean_question)


if __name__ == "__main__":
    main()
