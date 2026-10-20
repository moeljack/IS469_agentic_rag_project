"""
Chunking helpers: readable tables and correct section headings for IRAS-style PDFs.
This module optimises document layout and chunk serialization before vectorization:
"""
import re

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, TextItem

ANNEX = re.compile(r"^Annex \d+[A-Z]?\s*[-–—]")   # "Annex 1 - ...", "Annex 2B – ..."
LABEL_ONLY_HEADINGS = {"note", "notes"}           # bare labels Docling mistakes for section headings


class MarkdownTableProvider(ChunkingSerializerProvider):
    """Serialize tables as Markdown rows under their header, not as `col, row = value` triplets."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def build_chunker(tokenizer):
    return HybridChunker(tokenizer=tokenizer, merge_peers=True, serializer_provider=MarkdownTableProvider())


def fix_structure(doc):
    """Repair what Docling's layout model gets wrong, in place, before chunking.

    1. A table caption such as "Annex 1 - ..." is labelled `caption` and glued to the previous section.
       Turn it into a real heading placed just before its table.
    2. A bare "Note:" is labelled a section header, so it replaces the real heading above it.
       Demote it to plain text.
    """
    for table in list(doc.tables):
        for cref in list(table.captions):
            caption = cref.resolve(doc)
            if ANNEX.match(caption.text):
                doc.insert_heading(sibling=table, text=caption.text, orig=caption.orig, level=1,
                                   prov=caption.prov[0] if caption.prov else None, after=False)
                table.captions = [c for c in table.captions if c.cref != cref.cref]
                table.children = [c for c in table.children if c.cref != cref.cref]

    for i, item in enumerate(doc.texts):
        if item.label == DocItemLabel.SECTION_HEADER and item.text.strip().rstrip(":").lower() in LABEL_ONLY_HEADINGS:
            doc.texts[i] = TextItem(self_ref=item.self_ref, parent=item.parent, children=item.children,
                                    content_layer=item.content_layer, label=DocItemLabel.TEXT,
                                    prov=item.prov, orig=item.orig, text=item.text)
    return doc
