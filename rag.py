from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import faiss

import models

DEFAULT_DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))
DEFAULT_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "index"))
INDEX_FILE_NAME = "faiss.index"
METADATA_FILE_NAME = "metadata.json"

_PROMPT_TEMPLATE = """あなたは講義資料に基づいて答えるアシスタントです。
回答は日本語で、必ず引用（最大3件）を併記してください。
質問: {question}

参考資料（抜粋）:
{context_blocks}

ルール:

資料にないことは「わからない」と述べる

引用には出典とposを付与（例: {{source}} / pos={{pos}}）

用語は簡潔に説明し、必要なら短い注釈を添える
"""


@dataclass
class Chunk:
    text: str
    source: str
    pos: int
    document_path: str


_index: faiss.Index | None = None
_metadata: dict[str, Any] | None = None
_index_dir: Path = DEFAULT_INDEX_DIR


def _get_index_path(index_dir: Path | None = None) -> Path:
    base = Path(index_dir) if index_dir else _index_dir
    return base / INDEX_FILE_NAME


def _get_metadata_path(index_dir: Path | None = None) -> Path:
    base = Path(index_dir) if index_dir else _index_dir
    return base / METADATA_FILE_NAME


def index_exists(index_dir: Path | None = None) -> bool:
    index_path = _get_index_path(index_dir)
    metadata_path = _get_metadata_path(index_dir)
    return index_path.exists() and metadata_path.exists()


def _reset_cache() -> None:
    global _index, _metadata
    _index = None
    _metadata = None


def _load_metadata(index_dir: Path | None = None) -> dict[str, Any] | None:
    global _metadata
    if _metadata is not None:
        return _metadata

    metadata_path = _get_metadata_path(index_dir)
    if not metadata_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as f:
        _metadata = json.load(f)
    return _metadata


def _load_index(index_dir: Path | None = None) -> faiss.Index | None:
    global _index
    if _index is not None:
        return _index

    index_path = _get_index_path(index_dir)
    if not index_path.exists():
        return None

    _index = faiss.read_index(str(index_path))
    return _index


def build_index(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    """Read markdown files, build embeddings, and persist the FAISS index."""

    data_path = Path(data_dir)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    documents = list(sorted(data_path.glob("*.md")))
    if not documents:
        raise FileNotFoundError(f"Markdownファイルが {data_path} に見つかりません。")

    chunks: list[Chunk] = []
    for doc in documents:
        content = doc.read_text(encoding="utf-8")
        metadata, body = _split_front_matter(content)
        source = metadata.get("source") or doc.stem
        for idx, chunk_text in enumerate(
            _chunk_text(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap), start=1
        ):
            cleaned = chunk_text.strip()
            if not cleaned:
                continue
            chunks.append(
                Chunk(
                    text=cleaned,
                    source=source,
                    pos=idx,
                    document_path=str(doc.relative_to(data_path.parent)),
                )
            )

    if not chunks:
        raise ValueError("チャンクが生成されませんでした。入力データを確認してください。")

    embeddings = models.embed_texts([chunk.text for chunk in chunks])
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("埋め込み数とチャンク数が一致しません。")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(_get_index_path(index_path)))

    metadata = {
        "chunks": [
            {
                "text": chunk.text,
                "source": chunk.source,
                "pos": chunk.pos,
                "document_path": chunk.document_path,
            }
            for chunk in chunks
        ],
        "stats": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "built_at": datetime.utcnow().isoformat() + "Z",
            "data_dir": str(data_path),
            "index_dir": str(index_path),
        },
    }

    with _get_metadata_path(index_path).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    global _index, _metadata, _index_dir
    _index = index
    _metadata = metadata
    _index_dir = index_path

    return metadata["stats"].copy()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        yield text[start:end]
        start += step


def _split_front_matter(content: str) -> tuple[dict[str, Any], str]:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            front_matter = parts[1]
            body = parts[2]
            metadata = _parse_yaml_like(front_matter)
            return metadata, body
    return {}, content


def _parse_yaml_like(fragment: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in fragment.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"")
    return metadata


def _ensure_index_loaded() -> bool:
    if _load_index() is None:
        return False
    if _load_metadata() is None:
        return False
    return True


def search(query: str, k: int = 4, score_threshold: float = 0.2) -> list[dict[str, Any]]:
    """Return top-k results from the FAISS index."""

    if not query.strip():
        return []

    if not _ensure_index_loaded():
        return []

    query_embedding = models.embed_texts([query])
    if query_embedding.size == 0:
        return []

    index = _load_index()
    assert index is not None

    scores, indices = index.search(query_embedding, k)
    metadata = _load_metadata()
    assert metadata is not None
    chunks = metadata["chunks"]

    results: list[dict[str, Any]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if float(score) < score_threshold:
            continue
        chunk = chunks[idx]
        result = {
            "text": chunk["text"],
            "source": chunk["source"],
            "pos": chunk["pos"],
            "document_path": chunk["document_path"],
            "score": float(score),
        }
        results.append(result)
    return results


def format_context(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, block in enumerate(blocks, start=1):
        snippet = block.get("text", "").replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(
            f"[{idx}] source={block.get('source', 'unknown')} pos={block.get('pos', '?')} :: {snippet}"
        )
    return "\n".join(lines)


def generate_answer(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = contexts[:4]
    if not relevant:
        return {
            "answer": "資料から該当情報を見つけられませんでした。わからないとお答えします。",
            "citations": [],
        }

    context_str = format_context(relevant)
    prompt = _PROMPT_TEMPLATE.format(question=question, context_blocks=context_str)
    answer = models.llm_answer(prompt)
    answer = answer.strip() if answer else "資料に基づいた回答を生成できませんでした。"

    citations: list[dict[str, Any]] = []
    for block in relevant[:3]:
        snippet = block.get("text", "").strip().replace("\n", " ")
        snippet = snippet[:160] + "…" if len(snippet) > 160 else snippet
        citations.append(
            {
                "source": block.get("source", "unknown"),
                "pos": block.get("pos", "?"),
                "snippet": snippet,
            }
        )

    return {"answer": answer, "citations": citations}


def get_index_stats(index_dir: str | Path | None = None) -> dict[str, Any]:
    metadata = _load_metadata(index_dir)
    if not metadata:
        return {
            "document_count": 0,
            "chunk_count": 0,
            "chunk_size": None,
            "chunk_overlap": None,
            "built_at": None,
        }
    stats = metadata.get("stats", {})
    return {
        "document_count": stats.get("document_count", 0),
        "chunk_count": stats.get("chunk_count", 0),
        "chunk_size": stats.get("chunk_size"),
        "chunk_overlap": stats.get("chunk_overlap"),
        "built_at": stats.get("built_at"),
    }
