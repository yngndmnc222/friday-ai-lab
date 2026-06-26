"""Chunk preprocessed text and call an embeddings model.

The database layer can store the records returned by ``chunk_and_embed`` once it
is ready. Until then, this module only prepares chunk text plus embeddings.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import requests


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_CHUNK_WORDS = 700
DEFAULT_CHUNK_OVERLAP_WORDS = 100


class EmbeddingError(RuntimeError):
    """Raised when an embeddings request cannot be completed."""


@dataclass(frozen=True)
class TextChunk:
    """A segment of text ready to be embedded."""

    id: str
    text: str
    start_word: int
    end_word: int
    word_count: int
    source: str | None = None
    metadata: dict[str, Any] | None = None

    def to_record(self, embedding: list[float]) -> dict[str, Any]:
        record = asdict(self)
        record["embedding"] = embedding
        return record


def coerce_preprocessed_text(data: Any) -> str:
    """Convert preprocessed input data into a single text string."""

    if data is None:
        return ""

    if isinstance(data, str):
        return data

    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")

    if isinstance(data, dict):
        parts: list[str] = []
        for key, value in data.items():
            value_text = coerce_preprocessed_text(value)
            if value_text:
                parts.append(f"{key}: {value_text}")
        return "\n".join(parts)

    if isinstance(data, Iterable):
        return "\n".join(
            part
            for part in (coerce_preprocessed_text(item) for item in data)
            if part
        )

    return str(data)


def chunk_text(
    text: str,
    *,
    max_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Split text into overlapping word chunks.

    Word-based chunking keeps this dependency-free. Tune ``max_words`` and
    ``overlap_words`` later if the selected model or data shape needs it.
    """

    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")

    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    words = normalized_text.split(" ")
    step = max_words - overlap_words
    chunks: list[TextChunk] = []

    for start_word in range(0, len(words), step):
        end_word = min(start_word + max_words, len(words))
        chunk_words = words[start_word:end_word]
        chunk_index = len(chunks)
        chunks.append(
            TextChunk(
                id=f"chunk-{chunk_index:04d}",
                text=" ".join(chunk_words),
                start_word=start_word,
                end_word=end_word,
                word_count=len(chunk_words),
                source=source,
                metadata=metadata.copy() if metadata else None,
            )
        )

        if end_word == len(words):
            break

    return chunks


class EmbeddingClient:
    """Small HTTP client for an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str | None = None,
        timeout_seconds: float = 30.0,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = (
            model
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self.api_url = (
            api_url
            or os.getenv("OPENAI_EMBEDDINGS_URL")
            or DEFAULT_EMBEDDINGS_URL
        )
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts and preserve input order."""

        clean_texts = [text.strip() for text in texts]
        if not clean_texts:
            return []

        if any(not text for text in clean_texts):
            raise ValueError("texts cannot contain empty strings")

        if not self.api_key:
            raise EmbeddingError("OPENAI_API_KEY is required to call embeddings")

        embeddings: list[list[float]] = []
        for start_index in range(0, len(clean_texts), self.batch_size):
            batch = clean_texts[start_index : start_index + self.batch_size]
            embeddings.extend(self._embed_batch(batch))

        return embeddings

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
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
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        try:
            body = response.json()
            data = sorted(body["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding response did not match expected shape") from exc


def chunk_preprocessed_data(
    preprocessed_data: Any,
    *,
    max_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Coerce and chunk already-preprocessed data."""

    text = coerce_preprocessed_text(preprocessed_data)
    return chunk_text(
        text,
        max_words=max_words,
        overlap_words=overlap_words,
        source=source,
        metadata=metadata,
    )


def chunk_and_embed(
    preprocessed_data: Any,
    *,
    client: EmbeddingClient | None = None,
    max_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return chunk records containing text, metadata, and embeddings."""

    chunks = chunk_preprocessed_data(
        preprocessed_data,
        max_words=max_words,
        overlap_words=overlap_words,
        source=source,
        metadata=metadata,
    )
    if not chunks:
        return []

    embedding_client = client or EmbeddingClient()
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise EmbeddingError("Embedding count did not match chunk count")

    return [
        chunk.to_record(embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
