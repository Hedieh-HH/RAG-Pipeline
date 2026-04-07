"""
Tests for Chunker.

Pure function tests — no credentials required, always run:
    python -m unittest tests/chunking/test_chunker.py -v

Integration tests require real Azure credentials and RUN_INTEGRATION=1:
    RUN_INTEGRATION=1 python -m unittest tests/chunking/test_chunker.py -v
"""

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragcore.chunking.chunker import (
    Chunker,
    _normalize_heading,
    _split_sentences,
    _count_tokens,
)
from ragcore.models import Chunk, ParsedDocument, ParsedElement, Section


def setUpModule():
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )


# Helpers — build minimal test objects without Azure SDK dependencies
def make_element(content: str, kind: str = "paragraph") -> ParsedElement:
    return ParsedElement(type=kind, content=content, bbox=[], offset=0, index=0)


def make_section(heading: str, elements: list[ParsedElement]) -> Section:
    return Section(heading=heading, level=1, elements=elements)


def make_doc(sections: list[Section]) -> ParsedDocument:
    return ParsedDocument(
        doc_id="test.pdf",
        doc_title="Test Document",
        source_blob="container/test.pdf",
        etag="test-etag",
        local_path=Path("/tmp/test.pdf"),
        page_count=1,
        sections=sections,
    )


# Chunker.chunk — basic behaviour
class TestChunkerBasic(unittest.TestCase):

    def test_empty_sections_returns_empty(self) -> None:
        doc = make_doc([])
        chunks = Chunker().chunk(doc)
        self.assertEqual(chunks, [])

    def test_empty_section_elements_returns_empty(self) -> None:
        doc = make_doc([make_section("Intro", [])])
        chunks = Chunker().chunk(doc)
        self.assertEqual(chunks, [])

    def test_returns_list_of_chunks(self) -> None:
        doc = make_doc([make_section("Intro", [make_element("Hello world.")])])
        chunks = Chunker().chunk(doc)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_chunk_content_is_non_empty(self) -> None:
        doc = make_doc([make_section("Intro", [make_element("Hello world.")])])
        chunks = Chunker().chunk(doc)
        for chunk in chunks:
            self.assertTrue(chunk.content.strip())

    def test_footnotes_are_dropped(self) -> None:
        elements = [
            make_element("Main content.", "paragraph"),
            make_element("This is a footnote.", "footnote"),
        ]
        doc = make_doc([make_section("Intro", elements)])
        chunks = Chunker().chunk(doc)
        for chunk in chunks:
            self.assertNotIn("footnote", chunk.content)

    def test_table_is_standalone_chunk(self) -> None:
        elements = [
            make_element("Before table.", "paragraph"),
            make_element("| A | B |\n| --- | --- |\n| 1 | 2 |", "table"),
            make_element("After table.", "paragraph"),
        ]
        doc = make_doc([make_section("Results", elements)])
        chunks = Chunker().chunk(doc)
        table_chunks = [c for c in chunks if "| --- |" in c.content]
        self.assertEqual(len(table_chunks), 1)
        # Table chunk should not contain surrounding paragraph text.
        self.assertNotIn("Before table", table_chunks[0].content)
        self.assertNotIn("After table", table_chunks[0].content)


# Chunker.chunk — chunk_index, chunk_total, prev/next links
class TestChunkerLinks(unittest.TestCase):

    def _get_chunks(self) -> list[Chunk]:
        elements = [make_element(f"Sentence number {i}. ") for i in range(20)]
        doc = make_doc([make_section("Section", elements)])
        return Chunker(chunk_size_tokens=20, overlap_tokens=0).chunk(doc)

    def test_chunk_index_is_sequential(self) -> None:
        chunks = self._get_chunks()
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata.chunk_index, i)

    def test_chunk_total_is_consistent(self) -> None:
        chunks = self._get_chunks()
        total = len(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.metadata.chunk_total, total)

    def test_first_chunk_has_no_prev(self) -> None:
        chunks = self._get_chunks()
        self.assertIsNone(chunks[0].prev_chunk_id)

    def test_last_chunk_has_no_next(self) -> None:
        chunks = self._get_chunks()
        self.assertIsNone(chunks[-1].next_chunk_id)

    def test_prev_next_links_are_consistent(self) -> None:
        chunks = self._get_chunks()
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i].next_chunk_id, chunks[i + 1].id)
            self.assertEqual(chunks[i + 1].prev_chunk_id, chunks[i].id)


# Chunker.chunk — chunk_size_tokens=None (no windowing)
class TestChunkerNoWindowing(unittest.TestCase):

    def test_one_chunk_per_section(self) -> None:
        sections = [
            make_section("A", [make_element("Content A.")]),
            make_section("B", [make_element("Content B.")]),
            make_section("C", [make_element("Content C.")]),
        ]
        doc = make_doc(sections)
        chunks = Chunker(chunk_size_tokens=None).chunk(doc)
        self.assertEqual(len(chunks), 3)

    def test_all_content_preserved(self) -> None:
        elements = [make_element(f"Para {i}.") for i in range(5)]
        doc = make_doc([make_section("Sec", elements)])
        chunks = Chunker(chunk_size_tokens=None).chunk(doc)
        self.assertEqual(len(chunks), 1)
        for i in range(5):
            self.assertIn(f"Para {i}", chunks[0].content)


# Chunker.chunk — section_heading normalisation
class TestChunkerHeadingNormalisation(unittest.TestCase):

    def test_numbered_heading_is_normalised(self) -> None:
        doc = make_doc([make_section("1.1 Background", [make_element("Text.")])])
        chunks = Chunker(chunk_size_tokens=None).chunk(doc)
        self.assertEqual(chunks[0].metadata.section_heading, "Background")

    def test_unnumbered_heading_unchanged(self) -> None:
        doc = make_doc([make_section("Background", [make_element("Text.")])])
        chunks = Chunker(chunk_size_tokens=None).chunk(doc)
        self.assertEqual(chunks[0].metadata.section_heading, "Background")


# Chunker._extract_overlap
class TestExtractOverlap(unittest.TestCase):

    def test_zero_overlap_returns_empty(self) -> None:
        chunker = Chunker(chunk_size_tokens=512, overlap_tokens=0)
        self.assertEqual(chunker._extract_overlap("Some text here."), "")

    def test_none_overlap_returns_empty(self) -> None:
        chunker = Chunker(chunk_size_tokens=512, overlap_tokens=None)
        self.assertEqual(chunker._extract_overlap("Some text here."), "")

    def test_short_text_returns_full_text(self) -> None:
        # Text shorter than overlap budget → return everything
        chunker = Chunker(chunk_size_tokens=512, overlap_tokens=100)
        short = "Hi."
        self.assertEqual(chunker._extract_overlap(short), short)

    def test_overlap_snaps_to_sentence_start(self) -> None:
        # Build text where the tail of the overlap window starts mid-sentence.
        # The returned overlap must start at a sentence boundary.
        chunker = Chunker(chunk_size_tokens=512, overlap_tokens=10)
        text = "First sentence here. Second sentence follows. Third sentence ends."
        overlap = chunker._extract_overlap(text)
        # Overlap must not start in the middle of a word / mid-sentence fragment.
        self.assertTrue(
            overlap[0].isupper() or not overlap,
            f"Overlap should start at a sentence boundary, got: {overlap!r}",
        )


# Oversized sentence — single sentence exceeding chunk_size_tokens
class TestOversizedSentence(unittest.TestCase):

    def test_oversized_sentence_emitted_as_standalone_chunk(self) -> None:
        # Build a sentence that is guaranteed to exceed a small chunk budget.
        long_sentence = " ".join(["word"] * 50)  # well over 20 tokens
        elements = [
            make_element("Short intro."),
            make_element(long_sentence),
        ]
        doc = make_doc([make_section("Section", elements)])
        chunks = Chunker(chunk_size_tokens=20, overlap_tokens=0).chunk(doc)
        # The long sentence must appear as its own chunk.
        oversized = [c for c in chunks if long_sentence in c.content]
        self.assertEqual(len(oversized), 1)

    def test_content_before_oversized_sentence_is_flushed(self) -> None:
        long_sentence = " ".join(["word"] * 50)
        elements = [
            make_element("Normal intro sentence."),
            make_element(long_sentence),
        ]
        doc = make_doc([make_section("Section", elements)])
        chunks = Chunker(chunk_size_tokens=20, overlap_tokens=0).chunk(doc)
        contents = [c.content for c in chunks]
        # Both the normal intro and the long sentence must be present.
        self.assertTrue(any("Normal intro" in c for c in contents))
        self.assertTrue(any(long_sentence in c for c in contents))


# Integration tests — skipped unless RUN_INTEGRATION=1
@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run")
class TestChunkerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from ragcore.config import DocIntelligenceSettings, LocalStorageSettings
        from ragcore.models import LocalDocument
        from ragcore.parsing.parser import DocumentIntelligenceParser

        doc_settings = DocIntelligenceSettings()  # type: ignore[call-arg]
        parser = DocumentIntelligenceParser(
            endpoint=doc_settings.endpoint,
            api_key=doc_settings.api_key,
            model_id=doc_settings.model_id,
        )

        pdf_dir = LocalStorageSettings().pdf_dir  # type: ignore[call-arg]
        pdfs = list(pdf_dir.glob("*.pdf"))
        assert pdfs, f"No PDFs found in {pdf_dir}"
        doc = LocalDocument(
            blob_name=pdfs[0].name,
            container_name="integration-test",
            etag="integration-test",
            local_path=pdfs[0].resolve(),
            sha256="0" * 64,
        )
        with parser:
            cls.parsed = parser.parse(doc)

        cls.chunks = Chunker().chunk(cls.parsed)

    def test_produces_chunks(self) -> None:
        self.assertGreater(len(self.chunks), 0)

    def test_all_chunks_have_content(self) -> None:
        for chunk in self.chunks:
            self.assertTrue(chunk.content.strip())

    def test_token_counts_are_positive(self) -> None:
        for chunk in self.chunks:
            self.assertGreater(chunk.token_count, 0)

    def test_chunk_index_and_total_consistent(self) -> None:
        total = len(self.chunks)
        for i, chunk in enumerate(self.chunks):
            self.assertEqual(chunk.metadata.chunk_index, i)
            self.assertEqual(chunk.metadata.chunk_total, total)

    def test_prev_next_links_valid(self) -> None:
        for i in range(len(self.chunks) - 1):
            self.assertEqual(self.chunks[i].next_chunk_id, self.chunks[i + 1].id)
            self.assertEqual(self.chunks[i + 1].prev_chunk_id, self.chunks[i].id)

    def test_doc_id_in_all_metadata(self) -> None:
        for chunk in self.chunks:
            self.assertEqual(chunk.metadata.doc_id, self.parsed.doc_id)


if __name__ == "__main__":
    unittest.main()
