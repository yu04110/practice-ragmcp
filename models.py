from __future__ import annotations

import json
import os
import re
import textwrap
from functools import lru_cache
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "sentence-transformers がインストールされていません。requirements.txt を参照してください。"
    ) from exc


DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence transformer model."""

    model_name = DEFAULT_EMBEDDING_MODEL
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings for the given texts."""

    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = _load_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    return embeddings


def llm_answer(prompt: str) -> str:
    """Generate an answer using an LLM or a deterministic fallback."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:  # pragma: no cover - depends on external service
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.responses.create(
                model=model_name,
                input=prompt,
            )
            content = response.output_text
            if content:
                return content.strip()
        except Exception:  # Fallback gracefully if the API call fails
            pass

    return _fallback_answer(prompt)


def _fallback_answer(prompt: str) -> str:
    """Generate a concise, citation-aware summary from the prompt."""

    question = _extract_question(prompt)
    contexts = _extract_contexts(prompt)

    if not contexts:
        return "資料から該当情報を見つけられませんでした。わからないとお答えします。"

    summaries: list[str] = []
    for ctx in contexts[:3]:
        snippet = re.sub(r"\s+", " ", ctx.get("text", "")).strip()
        snippet = textwrap.shorten(snippet, width=160, placeholder="…")
        source = ctx.get("source", "unknown")
        pos = ctx.get("pos", "?")
        summaries.append(f"{snippet}（{source} / pos={pos}）")

    if question:
        header = f"質問「{question}」に対する資料の要点です。"
    else:
        header = "資料の要点です。"

    body = "\n".join(f"- {summary}" for summary in summaries)
    return f"{header}\n{body}"


def _extract_question(prompt: str) -> str:
    match = re.search(r"質問:\s*(.+)", prompt)
    if match:
        return match.group(1).strip()
    return ""


def _extract_contexts(prompt: str) -> list[dict[str, Any]]:
    if "参考資料（抜粋）:" not in prompt:
        return []

    contexts_section = prompt.split("参考資料（抜粋）:", 1)[1]
    if "ルール:" in contexts_section:
        contexts_section = contexts_section.split("ルール:", 1)[0]

    contexts: list[dict[str, Any]] = []
    for line in contexts_section.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: [idx] source=... pos=... :: text
        match = re.match(
            r"\[(?P<idx>\d+)\]\s*source=(?P<source>[^\s]+)\s+pos=(?P<pos>[^\s]+)\s*::\s*(?P<text>.+)",
            line,
        )
        if match:
            contexts.append(match.groupdict())
        else:
            contexts.append({"text": line})
    return contexts
