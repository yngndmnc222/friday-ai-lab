"""LiteLLM/OpenAI-compatible chat client and retrieval helpers."""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import requests

from .api_urls import endpoint_url
from .embeddings import EmbeddingClient


DEFAULT_CHAT_API_BASE = "https://api.openai.com"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful citizen query resolution assistant. Answer clearly, "
    "ask for missing details when needed, and use the provided knowledge "
    "snippets only when they are relevant."
)


class ChatError(RuntimeError):
    """Raised when a chat request cannot be completed."""


class ChatClient:
    """Small requests-based client for a LiteLLM-compatible chat endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL")
        )
        configured_url = (
            api_url
            or os.getenv("LITELLM_API_URL")
            or os.getenv("LLM_CHAT_COMPLETIONS_URL")
            or os.getenv("OPENAI_CHAT_COMPLETIONS_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_CHAT_API_BASE
        )
        self.api_url = endpoint_url(configured_url, "chat/completions")
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the assistant response for a chat history."""

        if not messages:
            raise ValueError("messages cannot be empty")

        if not self.api_key:
            raise ChatError("LLM_API_KEY is required to chat")

        if not self.model:
            raise ChatError("LLM_MODEL is required to chat")

        payload = {
            "model": self.model,
            "messages": list(messages),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ChatError(f"Chat request failed: {exc}") from exc

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ChatError("Chat response did not match expected shape") from exc

        return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip()

    return str(content)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity for two embedding vectors."""

    if len(left) != len(right) or not left:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def rank_context_records(
    prompt: str,
    records: Sequence[dict[str, Any]],
    *,
    embedding_client: EmbeddingClient | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Rank embedded records by similarity to the current prompt."""

    if top_k <= 0 or not prompt.strip() or not records:
        return []

    client = embedding_client or EmbeddingClient()
    query_embedding = client.embed_texts([prompt])[0]
    scored_records: list[dict[str, Any]] = []

    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue

        score = cosine_similarity(query_embedding, embedding)
        scored_record = dict(record)
        scored_record["score"] = score
        scored_records.append(scored_record)

    return sorted(scored_records, key=lambda record: record["score"], reverse=True)[:top_k]


def build_context_message(records: Sequence[dict[str, Any]]) -> str:
    """Format retrieved chunks for the model."""

    if not records:
        return ""

    snippets = []
    for record in records:
        source = record.get("source") or "unknown source"
        snippets.append(
            f"[{record.get('id', 'chunk')} | source: {source}]\n{record.get('text', '')}"
        )

    return (
        "Relevant knowledge snippets:\n\n"
        + "\n\n".join(snippets)
        + "\n\nIf these snippets are not enough, say what information is missing."
    )


def build_chat_messages(
    history: Sequence[dict[str, str]],
    *,
    context_records: Sequence[dict[str, Any]] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history_limit: int = 12,
) -> list[dict[str, str]]:
    """Build model messages from app chat history and optional context."""

    messages = [{"role": "system", "content": system_prompt}]
    context_message = build_context_message(context_records or [])
    if context_message:
        messages.append({"role": "system", "content": context_message})

    for entry in history[-history_limit:]:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    return messages
