import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.embeddings import (
    EmbeddingError,
    chunk_and_embed,
    chunk_preprocessed_data,
    chunk_text,
    coerce_preprocessed_text,
)


class FakeEmbeddingClient:
    def embed_texts(self, texts):
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


class BrokenEmbeddingClient:
    def embed_texts(self, texts):
        return [[1.0]]


class EmbeddingPipelineTests(unittest.TestCase):
    def test_chunk_text_uses_overlap(self):
        chunks = chunk_text("one two three four five", max_words=3, overlap_words=1)

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three",
            "three four five",
        ])
        self.assertEqual(chunks[1].start_word, 2)
        self.assertEqual(chunks[1].end_word, 5)

    def test_chunk_preprocessed_data_accepts_structured_input(self):
        chunks = chunk_preprocessed_data(
            {"title": "Benefits", "body": ["water service", "tax support"]},
            max_words=4,
            overlap_words=0,
            source="fixture",
            metadata={"language": "en"},
        )

        self.assertEqual(chunks[0].source, "fixture")
        self.assertEqual(chunks[0].metadata, {"language": "en"})
        self.assertIn("title: Benefits", chunks[0].text)

    def test_coerce_preprocessed_text_filters_empty_parts(self):
        text = coerce_preprocessed_text(["alpha", None, "", {"beta": "gamma"}])

        self.assertEqual(text, "alpha\nbeta: gamma")

    def test_chunk_and_embed_returns_storage_ready_records(self):
        records = chunk_and_embed(
            "alpha beta gamma delta epsilon",
            client=FakeEmbeddingClient(),
            max_words=3,
            overlap_words=1,
            source="fixture.txt",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], "chunk-0000")
        self.assertEqual(records[0]["text"], "alpha beta gamma")
        self.assertEqual(records[0]["embedding"], [0.0, 16.0])
        self.assertEqual(records[1]["source"], "fixture.txt")

    def test_chunk_and_embed_rejects_embedding_count_mismatch(self):
        with self.assertRaises(EmbeddingError):
            chunk_and_embed(
                "alpha beta gamma delta epsilon",
                client=BrokenEmbeddingClient(),
                max_words=3,
                overlap_words=1,
            )


if __name__ == "__main__":
    unittest.main()
