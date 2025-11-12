from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import rag
from tools import clear_history, save_note

st.set_page_config(page_title="Lecture RAG Chat", page_icon="📘", layout="wide")

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
TOP_K = 4


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("last_citations", [])
    st.session_state.setdefault("last_question", None)
    st.session_state.setdefault("index_stats", rag.get_index_stats())


_init_session_state()


def _auto_build_index() -> None:
    if rag.index_exists():
        return
    if not any(DATA_DIR.glob("*.md")):
        return
    with st.spinner("初回インデックスを構築中です…"):
        try:
            stats = rag.build_index(DATA_DIR, INDEX_DIR)
            st.session_state["index_stats"] = stats
            st.success("インデックスを構築しました。")
        except Exception as exc:
            st.error(f"インデックス構築に失敗しました: {exc}")


_auto_build_index()

st.title("📘 講義テキストRAGチャット")


with st.sidebar:
    st.header("操作")
    if st.button("インデックス構築", use_container_width=True):
        with st.spinner("インデックスを再構築中です…"):
            try:
                stats = rag.build_index(DATA_DIR, INDEX_DIR)
                st.session_state["index_stats"] = stats
                st.success("インデックスを更新しました。")
            except Exception as exc:
                st.error(f"インデックス構築に失敗しました: {exc}")

    last_answer: str | None = st.session_state.get("last_answer")
    last_citations: list[dict[str, Any]] = st.session_state.get("last_citations", [])
    last_question: str | None = st.session_state.get("last_question")

    if st.button("メモ保存", use_container_width=True, disabled=not last_answer):
        note_title = last_question or "チャットメモ"
        citation_lines = [
            f"- {c['snippet']}（{c['source']} / pos={c['pos']}）" for c in last_citations[:3]
        ]
        citation_block = "\n".join(citation_lines)
        note_content = last_answer
        if citation_block:
            note_content = f"{last_answer}\n\n## 引用\n{citation_block}"
        path = save_note(note_title, note_content)
        st.success(f"メモを保存しました: {path}")

    if st.button("履歴クリア", use_container_width=True):
        clear_history()
        st.success("履歴をクリアしました。")

    st.divider()
    stats = rag.get_index_stats()
    st.caption("インデックス状況")
    st.write(f"- 文書数: {stats.get('document_count', 0)}")
    st.write(f"- チャンク数: {stats.get('chunk_count', 0)}")
    st.write(f"- 最終構築: {stats.get('built_at') or '未構築'}")
    st.write(f"- 検索k: {TOP_K}")

if not rag.index_exists():
    st.info("インデックスが未構築です。サイドバーのボタンで構築してください。")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("citations"):
            st.markdown("**引用**")
            for citation in message["citations"][:3]:
                st.markdown(
                    f"- 『{citation['snippet']}』（{citation['source']} / pos={citation['pos']}）"
                )

prompt = st.chat_input("質問を入力してください")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state["messages"].append(user_message)
    st.session_state["last_question"] = prompt

    with st.chat_message("user"):
        st.markdown(prompt)

    contexts = rag.search(prompt, k=TOP_K)
    result = rag.generate_answer(prompt, contexts)
    answer = result.get("answer", "")
    citations = result.get("citations", [])

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "citations": citations,
    }
    st.session_state["messages"].append(assistant_message)
    st.session_state["last_answer"] = answer
    st.session_state["last_citations"] = citations

    with st.chat_message("assistant"):
        st.markdown(answer)
        if citations:
            st.markdown("**引用**")
            for citation in citations:
                st.markdown(
                    f"- 『{citation['snippet']}』（{citation['source']} / pos={citation['pos']}）"
                )
