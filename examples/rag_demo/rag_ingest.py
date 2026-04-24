#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "qdrant-client",
#   "httpx",
#   "apple-fm-cli",
#   "pypdf",
#   "langchain-text-splitters",
# ]
# ///
"""
Ingest `sample_corpus/` into Qdrant: **hierarchical** + **contextual** chunks, then
512-d Apple embeddings (see `chunking.py`). Start Qdrant: `docker compose up -d`.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator
from pathlib import Path

from chunking import (
    LeafChunk,
    ParentBlock,
    build_contextual_embed_string,
    recursive_chunk_text,
    semantic_cluster_sentences,
    split_markdown_into_blocks,
    stable_leaf_id,
)
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import apple_fm_sdk as fm

SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> str:
    lines: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts = [f"{k.strip()}: {v.strip()}" for k, v in row.items() if v and str(v).strip()]
            if parts:
                lines.append(" | ".join(parts))
    return "\n\n".join(lines)


def read_pdf_parent_blocks(path: Path) -> list[ParentBlock]:
    reader = PdfReader(str(path))
    blocks: list[ParentBlock] = []
    for i, page in enumerate(reader.pages, start=1):
        t = (page.extract_text() or "").strip()
        if t:
            blocks.append(
                ParentBlock(
                    hierarchy_id=f"page:{i}",
                    display_title=f"Page {i}",
                    parent_text=t,
                    body=t,
                )
            )
    return blocks


def load_parent_blocks(path: Path) -> list[ParentBlock]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_parent_blocks(path)
    if ext == ".md":
        return split_markdown_into_blocks(read_text_file(path))
    if ext == ".txt":
        t = read_text_file(path)
        if not t.strip():
            return []
        return [
            ParentBlock(
                hierarchy_id="document:body",
                display_title="Document",
                parent_text=t,
                body=t,
            )
        ]
    if ext == ".csv":
        t = read_csv(path)
        if not t.strip():
            return []
        return [
            ParentBlock(
                hierarchy_id="document:body",
                display_title="Spreadsheet",
                parent_text=t,
                body=t,
            )
        ]
    raise ValueError(f"Unsupported: {ext}")


def iter_corpus_files(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p


def _unique_include_rel(path: Path, used: set[str]) -> str:
    parent = path.parent.name or "root"
    rel = f"include/{parent}/{path.name}"
    if rel not in used:
        used.add(rel)
        return rel
    n = 2
    while True:
        stem = f"{path.stem}_{n}{path.suffix}"
        rel = f"include/{parent}/{stem}"
        n += 1
        if rel not in used:
            used.add(rel)
            return rel


def iter_all_sources(
    corpus_root: Path,
    include_paths: list[Path],
) -> Iterator[tuple[Path, str]]:
    if corpus_root.is_dir():
        for p in iter_corpus_files(corpus_root):
            yield p, str(p.relative_to(corpus_root))
    used: set[str] = set()
    for raw in include_paths:
        p = raw.expanduser().resolve()
        if not p.is_file():
            print(f"Skip missing include: {p}")
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            print(f"Skip unsupported include: {p}")
            continue
        yield p, _unique_include_rel(p, used)


def document_to_leaf_chunks(
    rel: str,
    blocks: list[ParentBlock],
    *,
    chunk_size: int,
    overlap: int,
    semantic_within_block: bool,
    semantic_threshold: float,
    max_semantic_embeds_per_block: int,
    contextual_embed: bool,
) -> list[LeafChunk]:
    """Turn parent blocks into leaf chunks; embed text is either contextual or raw."""
    out: list[LeafChunk] = []
    for block in blocks:
        bodies: list[str]
        if semantic_within_block and len(block.body) > chunk_size * 2:
            bodies = semantic_cluster_sentences(
                block.body,
                embed=fm.get_sentence_embedding,
                similarity_threshold=semantic_threshold,
                min_chunk_chars=max(80, chunk_size // 4),
                max_embed_calls=max_semantic_embeds_per_block,
            )
        else:
            bodies = [block.body]
        b_idx = 0
        for btext in bodies:
            if not btext.strip():
                b_idx += 1
                continue
            sub = recursive_chunk_text(
                btext, chunk_size=chunk_size, chunk_overlap=overlap
            )
            for s_idx, ch in enumerate(sub):
                hpath = f"{rel}/{block.hierarchy_id}/b{b_idx}/s{s_idx}"
                if contextual_embed:
                    cfe = build_contextual_embed_string(
                        source=rel,
                        hierarchy_path=hpath,
                        parent_scope=block.parent_text,
                        raw_passage=ch,
                    )
                else:
                    cfe = ch
                out.append(
                    LeafChunk(
                        raw_passage=ch,
                        hierarchy_path=hpath,
                        parent_scope=block.parent_text,
                        context_for_embedding=cfe,
                    )
                )
            b_idx += 1
    return out


def build_points(
    sources: list[tuple[Path, str]],
    *,
    chunk_size: int,
    overlap: int,
    semantic_within_block: bool,
    semantic_threshold: float,
    max_semantic_embeds: int,
    contextual_embed: bool,
) -> tuple[list[PointStruct], int]:
    points: list[PointStruct] = []
    for path, rel in sources:
        try:
            blocks = load_parent_blocks(path)
        except Exception as e:
            print(f"Skip {rel}: {e}")
            continue
        if not blocks:
            print(f"Skip (empty): {rel}")
            continue
        leaves = document_to_leaf_chunks(
            rel,
            blocks,
            chunk_size=chunk_size,
            overlap=overlap,
            semantic_within_block=semantic_within_block,
            semantic_threshold=semantic_threshold,
            max_semantic_embeds_per_block=max_semantic_embeds,
            contextual_embed=contextual_embed,
        )
        for leaf in leaves:
            vec = fm.get_sentence_embedding(leaf.context_for_embedding)
            pid = stable_leaf_id(rel, leaf.hierarchy_path)
            rp = leaf.raw_passage
            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "source": rel,
                        "hierarchy_path": leaf.hierarchy_path,
                        "text": rp,
                        "parent_scope": leaf.parent_scope[:3000]
                        if len(leaf.parent_scope) > 3000
                        else leaf.parent_scope,
                        "excerpt": rp[:280] + ("…" if len(rp) > 280 else ""),
                        "embedding_mode": "contextual" if contextual_embed else "raw",
                    },
                )
            )
    return points, len(points)


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Hierarchical, contextual chunking → embed → Qdrant."
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=root / "sample_corpus",
        help="Root folder to walk (default: sample_corpus/). If missing, --include is required.",
    )
    ap.add_argument(
        "--include",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Extra file to ingest (repeatable)",
    )
    ap.add_argument("--host", default="localhost", help="Qdrant host")
    ap.add_argument("--port", type=int, default=6333, help="Qdrant port")
    ap.add_argument(
        "--collection",
        default="apple_fm_rag_corpus",
        help="Qdrant collection name",
    )
    ap.add_argument("--chunk-size", type=int, default=700, help="Target leaf size (chars)")
    ap.add_argument("--overlap", type=int, default=120, help="Leaf overlap (chars)")
    ap.add_argument(
        "--no-recreate",
        action="store_true",
        help="Append to collection instead of recreating (default: recreate)",
    )
    ap.add_argument(
        "--no-contextual-embed",
        action="store_true",
        help="Embed raw passages only (disables source/position/parent preface in vectors)",
    )
    ap.add_argument(
        "--semantic-within-block",
        action="store_true",
        help=(
            "Within each page/section, cluster sentences by embedding similarity before "
            "recursive leafing (adds many embedding calls for long blocks)"
        ),
    )
    ap.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.55,
        help="Min cosine sim to stay in the same cluster (with --semantic-within-block)",
    )
    ap.add_argument(
        "--max-semantic-embeds-per-block",
        type=int,
        default=200,
        metavar="N",
        help="Cap per parent block (page/section) for semantic gating (safety bound)",
    )
    args = ap.parse_args()
    recreate = not args.no_recreate
    contextual = not args.no_contextual_embed

    client = QdrantClient(host=args.host, port=args.port)
    if recreate:
        print(
            f"Recreating collection {args.collection!r} (512-d cosine, hierarchical+context)..."
        )
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    corpus_root = args.corpus.resolve()
    if not corpus_root.is_dir() and not args.include:
        raise SystemExit(f"Corpus not found: {corpus_root}")

    includes = [Path(p) for p in args.include]
    sources = list(iter_all_sources(corpus_root, includes))
    if not sources:
        raise SystemExit("No input files. Check --corpus and --include paths.")

    if corpus_root.is_dir():
        print(f"Loading from {corpus_root}…")
    if includes:
        print("Additional includes:")
        for p in includes:
            print(f"  {p.resolve()}")
    if args.semantic_within_block:
        print(
            "Semantic gating: ON (threshold="
            f"{args.semantic_threshold}, max={args.max_semantic_embeds_per_block} embeds/block)"
        )
    print(
        f"Contextual embedding: {'ON' if contextual else 'OFF (raw only)'}",
    )
    points, n = build_points(
        sources,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        semantic_within_block=args.semantic_within_block,
        semantic_threshold=args.semantic_threshold,
        max_semantic_embeds=args.max_semantic_embeds_per_block,
        contextual_embed=contextual,
    )
    if n == 0:
        raise SystemExit("No points produced.")

    print(f"Upserting {n} points…")
    client.upsert(collection_name=args.collection, points=points)
    print("Done. Use: uv run rag_ask.py \"…\"")


if __name__ == "__main__":
    main()
