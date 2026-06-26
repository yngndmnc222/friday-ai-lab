from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings import (  # noqa: E402
    EmbeddingClient,
    EmbeddingError,
    chunk_and_embed,
    chunk_preprocessed_data,
)


def load_local_env() -> None:
    for env_path in (REPO_ROOT / ".env", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("embedding_records", [])
    st.session_state.setdefault("chunk_preview", [])


def uploaded_data_to_input(uploaded_file: Any) -> Any:
    if uploaded_file is None:
        return None

    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if uploaded_file.name.lower().endswith(".json"):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    return raw_text


def records_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        embedding = record.get("embedding") or []
        rows.append(
            {
                "id": record["id"],
                "words": record["word_count"],
                "start": record["start_word"],
                "end": record["end_word"],
                "embedding_dim": len(embedding),
                "source": record.get("source") or "",
                "text": record["text"][:160],
            }
        )
    return rows


def render_chat() -> None:
    st.subheader("Citizen Query Resolution Chatbot")

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    prompt = st.chat_input("Ask a public service question")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    embedded_chunks = len(st.session_state.embedding_records)
    response = (
        f"I received your question: {prompt}"
        if embedded_chunks == 0
        else f"I received your question and have {embedded_chunks} embedded chunks available."
    )
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_embeddings_workspace() -> None:
    st.subheader("Knowledge Embeddings")

    uploaded_file = st.file_uploader(
        "Preprocessed file",
        type=["txt", "md", "json", "csv"],
    )
    text_input = st.text_area(
        "Preprocessed text",
        height=220,
        placeholder="Paste cleaned text here...",
    )

    controls = st.columns([2, 1, 1])
    source = controls[0].text_input("Source label", value="")
    max_words = controls[1].number_input(
        "Chunk words",
        min_value=50,
        max_value=2000,
        value=700,
        step=50,
    )
    overlap_words = controls[2].number_input(
        "Overlap",
        min_value=0,
        max_value=max(0, int(max_words) - 1),
        value=min(100, max(0, int(max_words) - 1)),
        step=25,
    )

    preprocessed_data = uploaded_data_to_input(uploaded_file)
    if text_input.strip():
        preprocessed_data = text_input

    action_columns = st.columns([1, 1, 4])
    preview_clicked = action_columns[0].button("Preview Chunks", use_container_width=True)
    embed_clicked = action_columns[1].button("Generate Embeddings", use_container_width=True)

    metadata = {"source_filename": uploaded_file.name} if uploaded_file else None
    source_label = source.strip() or (uploaded_file.name if uploaded_file else None)

    if preview_clicked and preprocessed_data:
        st.session_state.chunk_preview = chunk_preprocessed_data(
            preprocessed_data,
            max_words=int(max_words),
            overlap_words=int(overlap_words),
            source=source_label,
            metadata=metadata,
        )
    elif preview_clicked:
        st.session_state.chunk_preview = []
        st.warning("Add preprocessed content before previewing chunks.")

    if embed_clicked and preprocessed_data:
        try:
            client = EmbeddingClient()
            with st.spinner("Generating embeddings..."):
                st.session_state.embedding_records = chunk_and_embed(
                    preprocessed_data,
                    client=client,
                    max_words=int(max_words),
                    overlap_words=int(overlap_words),
                    source=source_label,
                    metadata=metadata,
                )
            st.session_state.chunk_preview = []
            st.success(f"Generated {len(st.session_state.embedding_records)} embedded chunks.")
        except (EmbeddingError, ValueError) as exc:
            st.error(str(exc))
    elif embed_clicked:
        st.warning("Add preprocessed content before generating embeddings.")

    preview_chunks = st.session_state.chunk_preview
    if preview_chunks:
        st.caption(f"{len(preview_chunks)} chunks ready")
        st.dataframe(
            [
                {
                    "id": chunk.id,
                    "words": chunk.word_count,
                    "start": chunk.start_word,
                    "end": chunk.end_word,
                    "source": chunk.source or "",
                    "text": chunk.text[:160],
                }
                for chunk in preview_chunks
            ],
            use_container_width=True,
            hide_index=True,
        )

    records = st.session_state.embedding_records
    if records:
        dimensions = len(records[0].get("embedding", []))
        metric_columns = st.columns(3)
        metric_columns[0].metric("Chunks", len(records))
        metric_columns[1].metric("Dimensions", dimensions)
        metric_columns[2].metric("Words", sum(record["word_count"] for record in records))
        st.dataframe(records_table(records), use_container_width=True, hide_index=True)


def main() -> None:
    load_local_env()
    st.set_page_config(page_title="Citizen Query Resolution Chatbot", layout="wide")
    init_session_state()

    st.title("Citizen Query Resolution")

    chat_tab, embeddings_tab = st.tabs(["Chat", "Knowledge"])
    with chat_tab:
        render_chat()
    with embeddings_tab:
        render_embeddings_workspace()


if __name__ == "__main__":
    main()
