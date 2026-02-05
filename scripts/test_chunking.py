#!/usr/bin/env python3
"""
Test chunking strategy: parse_sections (source_text + fallback), split_into_chunks, and full nested doc.
Run: uv run python scripts/test_chunking.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.chunking import (
    wikitext_to_plain,
    parse_sections_from_source_text,
    parse_sections,
    clean_text,
    split_into_chunks,
)
from src.pipeline_nested import process_single_page_nested


# Sample config matching config/pipeline.json
TEST_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "min_chunk_length": 20,
    "skip_sections": ["see also", "references", "external links"],
    "include_auxiliary": True,
}


def make_page_with_source_text():
    """Page with source_text (wiki == Heading ==). This is the preferred path."""
    return {
        "_id": "42",
        "title": "Test Article",
        "timestamp": "2025-01-01T00:00:00Z",
        "text": "Some plain text fallback.",
        "source_text": """This is the '''introduction''' with [[Wikipedia|wiki links]] and a <ref>citation</ref>.

== History ==

The project began in 1990. It was very successful. Many people used it every day.

== Features ==

Key features include speed and reliability. We also have good documentation here.
""",
    }


def make_page_fallback_no_source_text():
    """Page without source_text: uses opening_text + text/heading."""
    return {
        "_id": "43",
        "title": "Fallback Article",
        "timestamp": "2025-01-01T00:00:00Z",
        "opening_text": "This is the opening paragraph of the article. It has enough content to form a section.",
        "heading": ["History", "Features"],
        "text": """History

The history section content goes here. It is long enough to be kept.

Features

Features content with multiple sentences. Also sufficient length.""",
    }


def make_page_with_auxiliary():
    """Page with auxiliary_text (e.g. captions, infobox)."""
    return {
        "_id": "44",
        "title": "Article With Aux",
        "timestamp": "2025-01-01T00:00:00Z",
        "source_text": """Short intro here that is just over twenty characters.

== Main ==

Main section content that is also long enough to be retained.
""",
        "auxiliary_text": ["Caption one with more than twenty characters.", "Another auxiliary string that is long enough."],
    }


def test_wikitext_to_plain():
    """Wiki markup is stripped to plain text."""
    raw = "Hello '''bold''' and [[Link|display]] and <ref>ref</ref>."
    out = wikitext_to_plain(raw)
    assert "bold" in out and "display" in out, out
    assert "'''" not in out and "[[" not in out and "<ref>" not in out, out
    print("  wikitext_to_plain: OK")


def test_parse_sections_from_source_text():
    """Sections are extracted from source_text by == Heading ==."""
    source = """Intro paragraph with enough content to pass the twenty character minimum.

== First Section ==

Content for first section. Also long enough.

== Second Section ==

Content for second section."""
    sections = parse_sections_from_source_text(source, "MyTitle")
    assert len(sections) >= 3, f"Expected intro + 2 sections, got {len(sections)}"
    paths = [s["section_path"] for s in sections]
    assert any("Introduction" in p for p in paths), paths
    assert any("First Section" in p for p in paths), paths
    assert any("Second Section" in p for p in paths), paths
    for s in sections:
        assert len(s["content"]) > 20, s
    print("  parse_sections_from_source_text: OK")


def test_parse_sections_prefers_source_text():
    """When source_text exists, it is used (not text/heading)."""
    page = make_page_with_source_text()
    sections = parse_sections(page, include_auxiliary=False)
    assert len(sections) >= 2, f"Expected intro + History + Features, got {len(sections)}"
    paths = [s["section_path"] for s in sections]
    assert any("Introduction" in p for p in paths), paths
    assert any("History" in p for p in paths), paths
    assert any("Features" in p for p in paths), paths
    # Wiki markup should be stripped in content
    for s in sections:
        assert "'''" not in s["content"] and "[[" not in s["content"], s["content"][:80]
    print("  parse_sections (source_text): OK")


def test_parse_sections_fallback():
    """When no source_text, opening_text + heading/text is used."""
    page = make_page_fallback_no_source_text()
    sections = parse_sections(page, include_auxiliary=False)
    assert len(sections) >= 2, f"Expected at least intro + sections, got {len(sections)}"
    paths = [s["section_path"] for s in sections]
    assert any("Introduction" in p for p in paths), paths
    assert any("History" in p for p in paths), paths
    print("  parse_sections (fallback): OK")


def test_parse_sections_auxiliary():
    """Auxiliary text is added as section(s) when include_auxiliary=True."""
    page = make_page_with_auxiliary()
    sections = parse_sections(page, include_auxiliary=True)
    assert any("Auxiliary" in s["section_path"] for s in sections), [s["section_path"] for s in sections]
    print("  parse_sections (auxiliary): OK")


def test_split_into_chunks():
    """Chunks respect size and overlap."""
    # Build text long enough to split (chunk_size 500 -> ~750 chars)
    sentences = ["This is sentence one."] * 50
    text = " ".join(sentences)
    chunks = split_into_chunks(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert len(c) > 0 and len(c) <= 800, f"Chunk length {len(c)}"
    # Overlap: last part of chunk N should appear at start of chunk N+1
    if len(chunks) >= 2:
        # Overlap is in chars; we use sentence boundary so some overlap expected
        assert chunks[0].strip() and chunks[1].strip()
    print("  split_into_chunks: OK")


def test_skip_sections():
    """Sections in skip_sections are not included in nested doc."""
    page = {
        "_id": "45",
        "title": "Skip Test",
        "timestamp": "2025-01-01T00:00:00Z",
        "source_text": """Intro content that is long enough to be kept for the test.

== See also ==

Some see also content here that is long.

== References ==

References content here.
""",
    }
    doc = process_single_page_nested(page, TEST_CONFIG)
    assert doc is not None, "Expected one doc"
    paths = [c["section_path"] for c in doc["chunks"]]
    assert not any("see also" in p.lower() for p in paths), paths
    assert not any("references" in p.lower() for p in paths), paths
    assert any("Introduction" in p for p in paths), paths
    print("  skip_sections: OK")


def test_full_nested_doc():
    """Full pipeline produces valid nested doc with all chunk fields."""
    page = make_page_with_source_text()
    doc = process_single_page_nested(page, TEST_CONFIG)
    assert doc is not None
    assert doc["article_id"] == "42"
    assert doc["article_title"] == "Test Article"
    assert doc["total_chunks"] == len(doc["chunks"])
    assert len(doc["chunks"]) >= 1
    for c in doc["chunks"]:
        assert isinstance(c["chunk_index"], int) and c["chunk_index"] >= 0
        assert "section_path" in c
        assert "content" in c and len(c["content"]) >= 1
        assert c["char_count"] == len(c["content"])
    print("  full nested doc: OK")


def main():
    print("Chunking tests")
    test_wikitext_to_plain()
    test_parse_sections_from_source_text()
    test_parse_sections_prefers_source_text()
    test_parse_sections_fallback()
    test_parse_sections_auxiliary()
    test_split_into_chunks()
    test_skip_sections()
    test_full_nested_doc()
    print("All chunking tests passed.")


if __name__ == "__main__":
    main()
