"""
Tests for DocumentIntelligenceParser.

Pure function tests — no credentials required, always run:
    python -m unittest tests/parsing/test_parser.py -v

Integration tests require real Azure credentials and RUN_INTEGRATION=1:
    RUN_INTEGRATION=1 python -m unittest tests/parsing/test_parser.py -v
"""

import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myrag.models import LocalDocument, ParsedDocument, ParsedElement, Section
from myrag.parsing.parser import (
    DocumentIntelligenceParser,
    _clean_text,
    _table_to_markdown,
)


# Minimal stubs for Azure SDK objects (no network calls)
class _Span:
    def __init__(self, offset: int = 0):
        self.offset = offset


class _Paragraph:
    def __init__(self, content: str, role: str | None = None, offset: int = 0):
        self.content = content
        self.role = role
        self.bounding_regions = []
        self.spans = [_Span(offset)]


class _Figure:
    def __init__(self, elements: list[str]):
        self.elements = elements  # e.g. ["/paragraphs/3"]


class _Cell:
    def __init__(self, elements: list[str] | None = None):
        self.elements = elements or []


class _Table:
    def __init__(self, cells: list[_Cell] | None = None, caption=None):
        self.cells = cells or []
        self.caption = caption


class _AnalyzeResult:
    def __init__(self, paragraphs=None, figures=None, tables=None):
        self.paragraphs = paragraphs or []
        self.figures = figures or []
        self.tables = tables or []


_PARSER = DocumentIntelligenceParser(
    endpoint="https://fake.cognitiveservices.azure.com/",
    api_key="fake-key",
)


# _clean_text
class TestCleanText(unittest.TestCase):

    def test_dehyphenates_soft_line_break(self) -> None:
        self.assertEqual(_clean_text("hyphen-\nated"), "hyphenated")

    def test_joins_newline_with_space(self) -> None:
        self.assertEqual(_clean_text("line one\nline two"), "line one line two")

    def test_collapses_double_spaces(self) -> None:
        self.assertEqual(_clean_text("hello  world"), "hello world")

    def test_strips_whitespace(self) -> None:
        self.assertEqual(_clean_text("  hello  "), "hello")

    def test_mid_word_hyphen_unchanged(self) -> None:
        self.assertEqual(_clean_text("well-known"), "well-known")

    def test_empty_string(self) -> None:
        self.assertEqual(_clean_text(""), "")


# _table_to_markdown
class TestTableToMarkdown(unittest.TestCase):

    def _make_cell(self, row: int, col: int, content: str, kind: str = "body"):
        """Build a minimal mock table cell."""

        class Cell:
            row_index = row
            column_index = col
            elements = []
            spans = []

        c = Cell()
        c.content = content
        c.kind = kind
        return c

    def _make_table(self, rows: int, cols: int, cells):
        class Table:
            row_count = rows
            column_count = cols
            bounding_regions = []
            caption = None

        t = Table()
        t.cells = cells
        return t

    def test_empty_table_returns_empty_string(self) -> None:
        class Table:
            cells = None
            row_count = 0
            column_count = 0

        self.assertEqual(_table_to_markdown(Table()), "")

    def test_simple_table_has_separator_row(self) -> None:
        cells = [
            self._make_cell(0, 0, "Name", kind="columnHeader"),
            self._make_cell(0, 1, "Value", kind="columnHeader"),
            self._make_cell(1, 0, "A"),
            self._make_cell(1, 1, "1"),
        ]
        table = self._make_table(2, 2, cells)
        md = _table_to_markdown(table)
        self.assertIn("| --- |", md)
        self.assertIn("Name", md)
        self.assertIn("Value", md)
        self.assertIn("A", md)

    def test_separator_after_header_row(self) -> None:
        cells = [
            self._make_cell(0, 0, "Col", kind="columnHeader"),
            self._make_cell(1, 0, "val"),
        ]
        table = self._make_table(2, 1, cells)
        lines = _table_to_markdown(table).splitlines()
        self.assertIn("| --- |", lines[1])


# FileNotFoundError guard
class TestParserGuard(unittest.TestCase):

    def test_file_not_found_raises_before_api_call(self) -> None:
        parser = DocumentIntelligenceParser(
            endpoint="https://fake.cognitiveservices.azure.com/",
            api_key="fake-key",
        )
        doc = LocalDocument(
            blob_name="missing.pdf",
            container_name="test",
            etag="x",
            local_path=Path("/nonexistent/missing.pdf"),
            sha256="a" * 64,
        )
        with parser:
            with self.assertRaises(FileNotFoundError):
                parser.parse(doc)


# Integration tests — skipped unless RUN_INTEGRATION=1
@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "Set RUN_INTEGRATION=1 to run")
class TestParserIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from myrag.config import DocIntelligenceSettings

        from myrag.config import LocalStorageSettings

        doc_settings = DocIntelligenceSettings()  # type: ignore[call-arg]
        cls.parser = DocumentIntelligenceParser(
            endpoint=doc_settings.endpoint,
            api_key=doc_settings.api_key,
            model_id=doc_settings.model_id,
        )
        cls.parser.__enter__()

        pdf_dir = LocalStorageSettings().pdf_dir  # type: ignore[call-arg]
        pdfs = list(pdf_dir.glob("*.pdf"))
        assert pdfs, f"No PDFs found in {pdf_dir}"
        cls.doc = LocalDocument(
            blob_name=pdfs[0].name,
            container_name="integration-test",
            etag="integration-test",
            local_path=pdfs[0].resolve(),
            sha256="0" * 64,
        )
        cls.result = cls.parser.parse(cls.doc)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.parser.__exit__(None, None, None)

    def test_returns_parsed_document(self) -> None:
        self.assertIsInstance(self.result, ParsedDocument)

    def test_page_count_positive(self) -> None:
        self.assertGreater(self.result.page_count, 0)

    def test_sections_non_empty(self) -> None:
        self.assertGreater(len(self.result.sections), 0)

    def test_all_sections_are_section_instances(self) -> None:
        for s in self.result.sections:
            self.assertIsInstance(s, Section)

    def test_all_section_elements_are_parsed_elements(self) -> None:
        for s in self.result.sections:
            for el in s.elements:
                self.assertIsInstance(el, ParsedElement)

    def test_no_heading_elements_in_section_elements(self) -> None:
        # Heading elements open sections — they should not appear inside elements list.
        for s in self.result.sections:
            for el in s.elements:
                self.assertNotEqual(el.type, "heading")

    def test_element_types_are_valid(self) -> None:
        valid_types = {"title", "paragraph", "footnote", "table"}
        for s in self.result.sections:
            for el in s.elements:
                self.assertIn(el.type, valid_types)

    def test_doc_id_matches_blob_name(self) -> None:
        self.assertEqual(self.result.doc_id, self.doc.blob_name)

    def test_source_blob_contains_container_and_name(self) -> None:
        self.assertEqual(
            self.result.source_blob,
            f"{self.doc.container_name}/{self.doc.blob_name}",
        )

    def test_no_soft_hyphen_line_breaks_in_elements(self) -> None:
        import re

        pattern = re.compile(r"\w-\n\w")
        for s in self.result.sections:
            for el in s.elements:
                self.assertIsNone(
                    pattern.search(el.content),
                    f"Element in section '{s.heading}' still contains soft hyphen break",
                )


if __name__ == "__main__":
    unittest.main()
