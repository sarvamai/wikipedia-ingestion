"""
Chunking: wiki markup -> plain text, section parsing (source_text with == Heading ==), split into chunks.
"""
import re
from typing import Dict, List

WIKI_HEADING_PATTERN = re.compile(r"^(=+)\s*(.+?)\s*=+\s*$", re.MULTILINE)


def wikitext_to_plain(wikitext: str) -> str:
    """
    Strip wiki markup to approximate plain text.
    Handles {{templates}}, [[links]], '''bold'', <ref>...</ref>, etc.
    """
    if not wikitext:
        return ""
    s = wikitext
    for _ in range(50):
        if "{{" not in s:
            break
        new_s = re.sub(r"\{\{[^{}]*\}\}", "", s)
        if new_s == s:
            break
        s = new_s

    def replace_link(m):
        inner = m.group(1)
        if "|" in inner:
            return inner.split("|")[-1].strip()
        return inner.split("/")[-1].strip() if "/" in inner else inner.strip()

    s = re.sub(r"\[\[([^\]|]+(?:\|[^\]]+)?)\]\]", replace_link, s)
    s = re.sub(r"'''(.+?)'''", r"\1", s)
    s = re.sub(r"''(.+?)''", r"\1", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<ref[^>]*\s*/>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\[https?://[^\]]+\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"\[citation needed\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"&[a-zA-Z]+;", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


def parse_sections_from_source_text(source_text: str, title: str) -> List[Dict]:
    """
    Split source_text by == Heading == and return sections with plain text.
    Intro = content before first heading; then one section per heading.
    """
    if not source_text or not source_text.strip():
        return []
    sections = []
    matches = list(WIKI_HEADING_PATTERN.finditer(source_text))
    if not matches:
        plain = wikitext_to_plain(source_text)
        if plain and len(plain) > 20:
            sections.append({"section_path": f"{title} > Introduction", "content": plain})
        return sections
    intro = source_text[: matches[0].start()].strip()
    if intro:
        plain_intro = wikitext_to_plain(intro)
        if plain_intro and len(plain_intro) > 20:
            sections.append({"section_path": f"{title} > Introduction", "content": plain_intro})
    for i, match in enumerate(matches):
        heading_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source_text)
        content = source_text[start:end].strip()
        if not content or len(content) < 20:
            continue
        plain = wikitext_to_plain(content)
        if plain and len(plain) > 20:
            sections.append({"section_path": f"{title} > {heading_title}", "content": plain})
    return sections


def parse_sections(page: Dict, include_auxiliary: bool = True) -> List[Dict]:
    """
    Parse page into sections. Prefers source_text (== Heading ==) when present;
    otherwise falls back to text + heading list.
    """
    title = page.get("title", "Unknown")
    text = page.get("text", "")
    headings = page.get("heading", [])
    opening_text = page.get("opening_text", "")
    auxiliary_text = page.get("auxiliary_text", [])
    source_text = page.get("source_text", "")
    sections = []

    if source_text and isinstance(source_text, str) and len(source_text.strip()) > 0:
        sections = parse_sections_from_source_text(source_text.strip(), title)
    else:
        if opening_text:
            sections.append({
                "section_path": f"{title} > Introduction",
                "content": opening_text.strip(),
            })
        if not headings:
            if not sections and text:
                sections.append({"section_path": title, "content": text.strip()})
        else:
            heading_patterns = [re.escape(h.strip()) for h in headings if h.strip()]
            if heading_patterns:
                combined_pattern = r"(?:^|\n)(" + "|".join(heading_patterns) + r")(?:\n|$)"
                matches = list(re.finditer(combined_pattern, text))
                if matches:
                    if not opening_text:
                        intro = text[: matches[0].start()].strip()
                        if intro and len(intro) > 50:
                            sections.append({"section_path": f"{title} > Introduction", "content": intro})
                    for i, match in enumerate(matches):
                        heading_name = match.group(1).strip()
                        start_pos = match.end()
                        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                        content = text[start_pos:end_pos].strip()
                        if content and len(content) > 20:
                            sections.append({"section_path": f"{title} > {heading_name}", "content": content})
                elif not sections and text:
                    sections.append({"section_path": title, "content": text.strip()})

    if include_auxiliary and auxiliary_text:
        useful_aux = [item for item in auxiliary_text if isinstance(item, str) and len(item) > 20]
        if useful_aux:
            current_batch = []
            current_len = 0
            batch_num = 1
            for item in useful_aux:
                if current_len + len(item) > 2000 and current_batch:
                    sections.append({
                        "section_path": f"{title} > Auxiliary Information {batch_num}" if batch_num > 1 else f"{title} > Auxiliary Information",
                        "content": "\n\n".join(current_batch),
                        "is_auxiliary": True,
                    })
                    batch_num += 1
                    current_batch = [item]
                    current_len = len(item)
                else:
                    current_batch.append(item)
                    current_len += len(item)
            if current_batch:
                sections.append({
                    "section_path": f"{title} > Auxiliary Information {batch_num}" if batch_num > 1 else f"{title} > Auxiliary Information",
                    "content": "\n\n".join(current_batch),
                    "is_auxiliary": True,
                })
    return sections


def clean_text(text: str) -> str:
    """Clean text for chunking."""
    if not text:
        return ""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        chunk_size: Max characters per chunk (default: 2000 ≈ 512 tokens for distilbert)
        overlap: Overlap in characters between chunks (default: 200)
    
    Recommended chunk_size by model:
        - msmarco-distilbert-base-tas-b (512 tokens): 2000 chars
        - embeddinggemma-300m (2048 tokens): 8000 chars
        - Azure text-embedding-3 (8192 tokens): 30000 chars
    """
    if not text or len(text) < 20:
        return []
    char_size = chunk_size  # Now directly in characters
    char_overlap = overlap
    if len(text) <= char_size:
        return [text]
    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk, current_len = [], 0
    for sentence in sentences:
        if current_len + len(sentence) <= char_size:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            overlap_sents, overlap_len = [], 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= char_overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sents + [sentence]
            current_len = sum(len(s) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    # Filter out empty or whitespace-only chunks
    return [c.strip() for c in chunks if c and c.strip()]
