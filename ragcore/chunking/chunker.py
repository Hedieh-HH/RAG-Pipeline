"""
Chunker — single chunking strategy combining structure-awareness and sentence integrity.

Algorithm:
  1. For each section, iterate over its ParsedElements in order.
  2. Non-table elements: split content into sentences, then accumulate sentences
     up to the token budget. A sentence that alone exceeds the budget is emitted
     as an oversized standalone chunk (never split mid-sentence).
  3. Table elements are always emitted as standalone chunks — flushing the
     current buffer before and resetting overlap after.
  4. Footnote elements are skipped entirely.
  5. After emitting a text chunk, a sentence-snapped overlap is extracted from
     the end and prepended to the next chunk within the same section.
     Overlap resets to empty after a table chunk boundary.
  6. Finalise: assign chunk_index, chunk_total, and wire prev/next links.

Token counting uses tiktoken cl100k_base (the encoding for text-embedding-3-large).

To produce one chunk per section (no windowing), pass chunk_size_tokens=None.
"""

import logging
import re
from uuid import UUID

import spacy
import tiktoken

from ragcore.models import Chunk, ChunkMetadata, ParsedDocument, ParsedElement, Section

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_START = re.compile(r"[.!?]\s+")
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _NLP


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_sentences(text: str) -> list[str]:
    return [sent.text.strip() for sent in _get_nlp()(text).sents if sent.text.strip()]


def _normalize_heading(heading: str) -> str:
    """Strip leading section numbers from a heading, e.g. '1.1 Background' → 'Background'."""
    return re.sub(r"^[\d.]+\s+", "", heading.strip()).strip()


def _element_pages(element: ParsedElement) -> list[int]:
    pages = []
    for br in element.bbox:
        page_num = getattr(br, "page_number", None)
        if page_num is not None:
            pages.append(page_num)
    return pages


def _pages_from_elements(elements: list[ParsedElement]) -> tuple[int, int]:
    all_pages: list[int] = []
    for el in elements:
        all_pages.extend(_element_pages(el))
    if not all_pages:
        return 1, 1
    return min(all_pages), max(all_pages)


class Chunker:
    """
    Sentence-aware, structure-respecting chunker.

    Args:
        chunk_size_tokens: Target maximum token count per chunk (default 512).
                           Pass None to keep each section as a single chunk.
        overlap_tokens: Tokens carried from the end of each text chunk into the
                        next, snapped to a sentence start (default 50).
                        Set to 0 or None to disable overlap. Ignored when
                        chunk_size_tokens is None.
    """

    def __init__(
        self,
        chunk_size_tokens: int | None = 512,
        overlap_tokens: int | None = 50,
    ) -> None:
        if chunk_size_tokens is not None and chunk_size_tokens <= 0:
            raise ValueError(
                f"chunk_size_tokens must be a positive integer, got {chunk_size_tokens}"
            )
        if overlap_tokens is not None and overlap_tokens < 0:
            raise ValueError(
                f"overlap_tokens must be non-negative, got {overlap_tokens}"
            )
        if (
            chunk_size_tokens is not None
            and overlap_tokens is not None
            and overlap_tokens >= chunk_size_tokens
        ):
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than "
                f"chunk_size_tokens ({chunk_size_tokens})"
            )
        self._chunk_size = chunk_size_tokens
        self._overlap = overlap_tokens

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        logger.debug(
            "Starting chunking for '%s' (%d sections)", doc.doc_id, len(doc.sections)
        )

        if not doc.sections:
            logger.warning(
                "Document '%s' has no sections — returning empty chunk list", doc.doc_id
            )
            return []

        raw: list[tuple[str, list[ParsedElement], str]] = []
        for section in doc.sections:
            raw.extend(self._chunk_section(section))

        if not raw:
            logger.warning(
                "Document '%s' produced no chunks — all sections may be empty",
                doc.doc_id,
            )
            return []

        chunks = self._finalise(raw, doc)
        logger.info(
            "Chunked '%s': %d sections → %d chunks",
            doc.doc_id,
            len(doc.sections),
            len(chunks),
        )
        return chunks

    def _chunk_section(
        self, section: Section, doc_id: str = ""
    ) -> list[tuple[str, list[ParsedElement], str]]:
        # No chunking — the whole section is a single chunk.
        if self._chunk_size is None:
            text = " ".join(el.content for el in section.elements)
            return (
                [(_normalize_heading(section.heading), list(section.elements), text)]
                if text.strip()
                else []
            )

        # Each entry is (section_heading, source_elements, chunk_text).
        # source_elements tracks which ParsedElements contributed sentences to this chunk —
        # an element can appear in multiple chunks if its sentences were split across a budget boundary.
        raw: list[tuple[str, list[ParsedElement], str]] = []

        buffer_elements: list[ParsedElement] = (
            []
        )  # Elements contributing to the current chunk.
        buffer_sentences: list[str] = []  # Text pieces accumulated so far.
        buffer_tokens: int = 0  # Token count of everything in buffer_sentences.
        overlap_text: str = ""  # Sentence-snapped tail of the last emitted chunk.

        def _emit() -> None:
            nonlocal overlap_text, buffer_elements, buffer_sentences, buffer_tokens
            if not buffer_sentences:
                return
            text = " ".join(buffer_sentences)
            raw.append(
                (_normalize_heading(section.heading), list(buffer_elements), text)
            )
            overlap_text = self._extract_overlap(text)
            buffer_elements.clear()
            buffer_sentences.clear()
            buffer_tokens = 0

        def _start_overlap() -> None:
            nonlocal buffer_sentences, buffer_tokens
            if overlap_text:
                buffer_sentences.append(overlap_text)
                buffer_tokens += _count_tokens(overlap_text)

        for el in section.elements:
            if el.type == "footnote":
                continue

            if el.type == "table":
                # Tables are always standalone — flush current chunk, emit
                # table on its own, then start the next chunk clean.
                _emit()
                raw.append((_normalize_heading(section.heading), [el], el.content))
                overlap_text = ""

            else:  # regular text
                try:
                    sentences = _split_sentences(el.content)
                except Exception:
                    logger.warning(
                        "spaCy sentence splitting failed for element in section '%s' — "
                        "falling back to naive split on sentence-ending punctuation.",
                        section.heading,
                    )
                    sentences = [
                        s.strip()
                        for s in re.split(r"(?<=[.!?])\s+", el.content)
                        if s.strip()
                    ]

                for sent in sentences:
                    sent_tokens = _count_tokens(sent)

                    if sent_tokens > self._chunk_size:
                        # Single sentence exceeds the budget — emit standalone.
                        logger.warning(
                            "Oversized sentence (%d tokens > chunk_size %d) in section '%s' — "
                            "emitting as standalone chunk.",
                            sent_tokens,
                            self._chunk_size,
                            section.heading,
                        )
                        _emit()
                        raw.append((_normalize_heading(section.heading), [el], sent))
                        overlap_text = ""
                    elif buffer_tokens + sent_tokens > self._chunk_size:
                        # Sentence doesn't fit — flush, carry overlap, add sentence.
                        _emit()
                        _start_overlap()
                        buffer_elements.append(el)
                        buffer_sentences.append(sent)
                        buffer_tokens += sent_tokens
                    else:
                        buffer_elements.append(el)
                        buffer_sentences.append(sent)
                        buffer_tokens += sent_tokens

        _emit()
        logger.debug("Section '%s' → %d chunks", section.heading, len(raw))
        return raw

    def _extract_overlap(self, text: str) -> str:
        """
        Take the last overlap_tokens tokens from text and snap forward to the
        first sentence start so the overlap never begins mid-sentence.
        """
        if not self._overlap:
            return ""
        tokens = _ENCODING.encode(text)
        if len(tokens) <= self._overlap:
            return text
        overlap_raw = _ENCODING.decode(tokens[-self._overlap :])
        match = _SENTENCE_START.search(overlap_raw)
        return overlap_raw[match.end() :] if match else overlap_raw

    def _finalise(
        self,
        raw: list[tuple[str, list[ParsedElement], str]],
        doc: ParsedDocument,
    ) -> list[Chunk]:
        total = len(raw)  # Total number of chunks in the document.
        chunks: list[Chunk] = []

        for idx, (section_heading, elements, text) in enumerate(raw):
            start_page, end_page = _pages_from_elements(elements)
            metadata = ChunkMetadata(
                doc_id=doc.doc_id,
                doc_title=doc.doc_title,
                source_blob=doc.source_blob,
                etag=doc.etag,
                chunk_index=idx,
                chunk_total=total,
                section_heading=section_heading,
                start_page=start_page,
                end_page=end_page,
            )

            prev_id: UUID | None = chunks[-1].id if chunks else None
            chunk = Chunk(
                content=text,
                token_count=_count_tokens(text),
                metadata=metadata,
                prev_chunk_id=prev_id,
            )

            if chunks:
                chunks[-1] = chunks[-1].model_copy(update={"next_chunk_id": chunk.id})

            chunks.append(chunk)

        return chunks
