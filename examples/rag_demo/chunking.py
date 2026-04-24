"""
Chunking for RAG: **hierarchical** structure + **contextual** embeddings.

**Hierarchical (structure-first)** — split order depends on source type:

- **PDF** — one *parent* scope per *page*; leaf chunks are produced only *inside* a page, so
  a chunk never spans two pages. Path looks like ``.../page:3/b0/s0``.
- **Markdown** — prefer splits at ``#`` / ``##`` / ``###`` headings; each heading block is
  a parent. Path: ``section:Title Here/leaf:1`` (title truncated for ids).
- **Text / CSV** — single parent ``document``; path ``document/leaf:k``.

**Contextual (embedding-aware, rule-based)** — the vector is computed from a short, explicit
  preamble (source id, position in hierarchy, and a truncated *parent* excerpt) plus the
  raw passage. That follows the idea of *contextual retrieval* (making each vector aware of
  *which document and where* the passage sits) without an extra LLM to rewrite every chunk.
  The raw ``Passage:`` is still stored for display and for the answer model.

**Optional semantic gating (within a parent only)** — if enabled, *sentences* (rough split) are
  embedded; when cosine similarity to the running cluster drops below a threshold, a new
  sub-chunk starts. This gives softer boundaries than character overlap alone, at extra
  embedding cost (bounded per parent).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paragraph → line → sentence-ish → word → char (unchanged for leaf splits).
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def normalize_chunking_input(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


def recursive_chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    if not text or not text.strip():
        return []
    normalized = normalize_chunking_input(text)
    if not normalized:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators if separators is not None else list(DEFAULT_SEPARATORS),
        length_function=len,
        keep_separator=False,
        is_separator_regex=False,
    )
    return [c for c in splitter.split_text(normalized) if c.strip()]


def _slug(s: str, max_len: int = 64) -> str:
    t = re.sub(r"\s+", "-", s.strip().lower())
    t = re.sub(r"[^a-z0-9._-]+", "", t)[:max_len]
    return t or "section"


# --- Hierarchical source units -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParentBlock:
    """One node in the hierarchy (page, markdown section, or whole document for CSV/txt)."""

    hierarchy_id: str
    """Path segment, e.g. ``page:2`` or ``section:product-overview``."""
    display_title: str
    parent_text: str
    """Full text of this block (context for the contextual prefix)."""
    body: str
    """Text to chunk; usually same as parent_text except for pre/post processing."""


_RE_MD_HEAD = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def split_markdown_into_blocks(text: str) -> list[ParentBlock]:
    """Split on ATX headings (#, ##, ###)."""
    text = normalize_chunking_input(text)
    if not text:
        return []
    matches = list(_RE_MD_HEAD.finditer(text))
    if not matches:
        return [
            ParentBlock(
                hierarchy_id="document:body",
                display_title="Document",
                parent_text=text,
                body=text,
            )
        ]
    blocks: list[ParentBlock] = []
    # Preamble before first heading
    first = matches[0]
    if first.start() > 0:
        preamble = text[: first.start()].strip()
        if preamble:
            blocks.append(
                ParentBlock(
                    hierarchy_id="document:preamble",
                    display_title="Preamble",
                    parent_text=preamble,
                    body=preamble,
                )
            )
    for i, m in enumerate(matches):
        title = m.group(2).strip()[:200]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        hid = f"section:{_slug(title)}"
        blocks.append(
            ParentBlock(
                hierarchy_id=hid,
                display_title=title,
                parent_text=body,
                body=body,
            )
        )
    if not blocks:
        return [
            ParentBlock(
                hierarchy_id="document:body",
                display_title="Document",
                parent_text=text,
                body=text,
            )
        ]
    return blocks


def _rough_sentences(text: str) -> list[str]:
    """Very light sentence tokenization; good enough to gate semantic clusters."""
    t = text.strip()
    if not t:
        return []
    # Split on . ? ! when followed by space or newline, but keep short lists intact.
    parts = re.split(r'(?<=[.!?？！])\s+', t)
    return [p for p in parts if p.strip()]


def semantic_cluster_sentences(
    text: str,
    *,
    embed: Callable[[str], Sequence[float]],
    similarity_threshold: float,
    min_chunk_chars: int,
    max_embed_calls: int,
) -> list[str]:
    """
    Group sentences by dropping similarity to a running *cluster mean* (first sentence of
    cluster). Yields 1+ strings, each not exceeding what recursive splitting would later
    handle; if embedding budget is hit, the remainder is one blob.
    """
    sents = _rough_sentences(text)
    if len(sents) <= 1 or len(sents) > max_embed_calls:
        return [text] if text.strip() else []

    from apple_fm_sdk import retrieval as r

    clusters: list[list[str]] = []
    current: list[str] = [sents[0]]
    v_prev = list(embed(sents[0]))
    n_calls = 1

    for s in sents[1:]:
        if n_calls >= max_embed_calls - 1:
            current.append(s)
            continue
        v_s = list(embed(s))
        n_calls += 1
        if r.cosine_similarity(v_prev, v_s) >= similarity_threshold and sum(
            len(x) for x in current
        ) + len(s) < 8000:
            current.append(s)
            v_prev = v_s
        else:
            clusters.append(current)
            current = [s]
            v_prev = v_s

    if current:
        clusters.append(current)

    out: list[str] = []
    for c in clusters:
        blob = " ".join(c).strip()
        if len(blob) < min_chunk_chars and out:
            out[-1] = (out[-1] + " " + blob).strip()
        else:
            out.append(blob)
    return [x for x in out if x]


@dataclass(frozen=True, slots=True)
class LeafChunk:
    raw_passage: str
    """Verbatim text for the LLM (no context boilerplate)."""
    hierarchy_path: str
    """e.g. ``Dimecraft.pdf/page:1/leaf:0`` relative to *rel* in ingest."""
    parent_scope: str
    context_for_embedding: str
    """Full string that was passed to the embedder."""


def build_contextual_embed_string(
    *,
    source: str,
    hierarchy_path: str,
    parent_scope: str,
    raw_passage: str,
    max_parent_chars: int = 650,
) -> str:
    """Rule-based contextual string (source + where + parent excerpt + passage)."""
    p = parent_scope.strip()
    if len(p) > max_parent_chars:
        p = p[: max_parent_chars - 1] + "…"
    return (
        f"Source: {source}\n"
        f"Position: {hierarchy_path}\n"
        f"Parent scope (excerpt from surrounding section or page):\n{p}\n\n"
        f"Passage to retrieve:\n{raw_passage.strip()}"
    )


def stable_leaf_id(rel: str, hierarchy_path: str) -> str:
    h = hashlib.sha256(f"{rel}::{hierarchy_path}".encode())
    return str(uuid.UUID(bytes=h.digest()[:16]))
