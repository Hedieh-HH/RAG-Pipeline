import logging
import re
import sys
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    DocumentTable,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError, HttpResponseError

from myrag.models import LocalDocument, ParsedDocument, ParsedElement, Section

logger = logging.getLogger(__name__)

# Azure SDK paragraph roles to skip entirely.
_SKIP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


class DocumentIntelligenceParser:
    """
    Context manager wrapping the Azure Document Intelligence SDK.

    Always use via `with`:

        with DocumentIntelligenceParser(endpoint, api_key) as parser:
            parsed_doc = parser.parse(local_document)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-layout",
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model_id = model_id
        self._client: DocumentIntelligenceClient | None = None

    def __enter__(self) -> "DocumentIntelligenceParser":
        self._client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._client:
            self._client.close()

    def _call_document_intelligence(self, document: LocalDocument) -> AnalyzeResult:
        """
        Read the local PDF and submit it to Document Intelligence.

        Raises:
            FileNotFoundError: if the local PDF does not exist.
            HttpResponseError: on API-level errors.
            AzureError: on general Azure connectivity issues.
        """
        if not document.local_path.exists():
            raise FileNotFoundError(
                f"PDF not found at expected path: {document.local_path}"
            )

        logger.info("Parsing '%s' with model '%s'", document.blob_name, self._model_id)

        try:
            with open(document.local_path, "rb") as f:
                file_bytes = f.read()
            poller = self._client.begin_analyze_document(
                self._model_id,
                AnalyzeDocumentRequest(bytes_source=file_bytes),
            )
            return poller.result()
        except HttpResponseError as e:
            logger.error(
                "Document Intelligence API error for '%s': %s", document.blob_name, e
            )
            raise
        except AzureError as e:
            logger.error("Azure error while parsing '%s': %s", document.blob_name, e)
            raise

    def _collect_removed_indices(self, result: AnalyzeResult) -> set[int]:
        """Collect paragraph indices to remove:
        - Headers, footers, and page numbers.
        - Texts overlaid on figures (e.g. chart labels, axis values).
        - Table captions and cell content (tables are processed separately as Markdown).

        Each elements entry is a string like "/paragraphs/5" (1-based) pointing to a
        paragraph by index in result.paragraphs.
        """
        removed: set[int] = set()

        for i, p in enumerate(result.paragraphs or []):
            if getattr(p, "role", None) in _SKIP_ROLES:
                removed.add(i)

        for figure in result.figures or []:
            for ref in getattr(figure, "elements", None) or []:
                try:
                    removed.add(int(ref.split("/")[-1]))
                except ValueError:
                    pass

        for table in result.tables or []:
            sources = list(table.cells or [])
            if getattr(table, "caption", None):
                sources.append(table.caption)
            for source in sources:
                for ref in getattr(source, "elements", None) or []:
                    try:
                        removed.add(int(ref.split("/")[-1]))
                    except ValueError:
                        pass

        return removed

    def _build_paragraph_elements(
        self, result: AnalyzeResult, removed_indices: set[int]
    ) -> list[ParsedElement]:
        """
        Build ParsedElement objects from result.paragraphs, skipping removed indices.

        Reference:
        https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0&tabs=rest%2Csample-code#sections:~:text=Sciences%20%7C%20Microsoft%22%0A%20%20%20%20%7D%0A%5D-,Paragraph%20roles,-The%20new%20page
        """
        elements: list[ParsedElement] = []

        for i, p in enumerate(result.paragraphs or []):
            if i in removed_indices:
                continue

            role = getattr(p, "role", None)
            content = _clean_text(p.content or "")
            bbox = p.bounding_regions or []
            offset = p.spans[0].offset if p.spans else 0

            if role == "title":
                kind = "title"
            elif role == "sectionHeading":
                kind = "heading"
            elif role == "footnote":
                kind = "footnote"
            else:
                kind = "paragraph"

            elements.append(
                ParsedElement(
                    type=kind,
                    content=content,
                    bbox=bbox,
                    offset=offset,
                    index=i,
                )
            )

        return elements

    def _build_table_elements(self, result: AnalyzeResult) -> list[ParsedElement]:
        """Convert each table in result.tables into a ParsedElement with Markdown content.

        Offset: minimum span offset across the caption and all cells.
        Bbox: taken from the table's own bounding_regions.
        """
        elements: list[ParsedElement] = []

        for table in result.tables or []:
            md = _table_to_markdown(table)
            if not md:
                continue

            caption = getattr(table, "caption", None)
            if caption and caption.content:
                md = caption.content.strip() + "\n\n" + md

            # Collect all paragraph indices referenced by caption and cells.
            para_indices: list[int] = []
            sources = list(table.cells or [])
            if getattr(table, "caption", None):
                sources.append(table.caption)
            for source in sources:
                for ref in getattr(source, "elements", None) or []:
                    try:
                        para_indices.append(int(ref.split("/")[-1]))
                    except ValueError:
                        pass

            offsets: list[int] = []
            if getattr(table, "caption", None) and table.caption.spans:
                offsets.append(table.caption.spans[0].offset)
            for cell in table.cells or []:
                if cell.spans:
                    offsets.append(cell.spans[0].offset)
            offset = min(offsets) if offsets else 0

            bbox = table.bounding_regions or []

            elements.append(
                ParsedElement(
                    type="table",
                    content=md,
                    bbox=bbox,
                    offset=offset,
                    index=para_indices,
                )
            )

        return elements

    def _sort_elements(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Sort ParsedElements in reading order based on their paragraph index.

        Paragraphs use their single int index directly.
        Tables use the minimum index among their referenced paragraphs.
        """

        def sort_key(element: ParsedElement) -> int:
            if isinstance(element.index, list):
                return min(element.index) if element.index else 0
            return element.index

        return sorted(elements, key=sort_key)

    def _build_sections(self, elements: list[ParsedElement]) -> list[Section]:
        """
        Build a list of Sections from a sorted list of ParsedElements.
        Each section starts at a heading element.
        All following elements belong to that section until the next heading.
        """
        sections: list[Section] = []
        current_section: Section | None = None

        for el in elements:
            if el.type == "heading":
                current_section = Section(
                    heading=el.content.strip(),
                    level=1,
                    elements=[],
                )
                sections.append(current_section)
            else:
                if current_section is None:
                    current_section = Section(
                        heading="__root__",
                        level=0,
                        elements=[],
                    )
                    sections.append(current_section)
                current_section.elements.append(el)

        return sections

    def parse(self, document: LocalDocument) -> ParsedDocument:
        """Submit a PDF to Document Intelligence and return a ParsedDocument."""
        # Step 1: call Azure Document Intelligence and get the raw result.
        result = self._call_document_intelligence(document)
        # Step 2: collect paragraph indices to remove (headers/footers, figures, tables).
        removed_indices = self._collect_removed_indices(result)
        # Step 3: build ParsedElements for paragraphs and tables separately.
        paragraph_elements = self._build_paragraph_elements(result, removed_indices)
        table_elements = self._build_table_elements(result)
        # Step 4: merge and sort all elements in reading order.
        elements = self._sort_elements(paragraph_elements + table_elements)
        # Step 5: group elements into sections.
        sections = self._build_sections(elements)
        # Step 6: build and return the ParsedDocument.
        doc_title = next((e.content for e in elements if e.type == "title"), None)
        page_count = len(result.pages) if result.pages else 1

        logger.info(
            "Parsed '%s': %d pages, %d sections",
            document.blob_name,
            page_count,
            len(sections),
        )

        return ParsedDocument(
            doc_id=document.blob_name,
            doc_title=doc_title,
            source_blob=f"{document.container_name}/{document.blob_name}",
            etag=document.etag,
            local_path=document.local_path,
            page_count=page_count,
            sections=sections,
        )


def _clean_text(text: str) -> str:
    """De-hyphenate soft line-breaks, join newlines, collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _table_to_markdown(table: DocumentTable) -> str:
    """Convert a DocumentTable to GitHub-Flavoured Markdown."""
    if not table.cells:
        return ""

    row_count = table.row_count
    col_count = table.column_count
    grid: list[list[str]] = [[""] * col_count for _ in range(row_count)]
    header_row: int | None = None

    for cell in table.cells:
        r, c = cell.row_index, cell.column_index
        grid[r][c] = (cell.content or "").replace("\n", " ").strip()
        if getattr(cell, "kind", None) == "columnHeader":
            header_row = r

    lines: list[str] = []
    for row_idx, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if row_idx == (header_row if header_row is not None else 0):
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines)
