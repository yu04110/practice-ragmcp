from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st


NOTES_DIR = Path("notes")


def save_note(title: str, content: str) -> str:
    """Save the given content as a markdown note and return the file path."""

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    slug = _slugify(title)
    filename = f"{timestamp}_{slug}.md"
    note_path = NOTES_DIR / filename
    note_body = f"# {title}\n\n{content.strip()}\n"
    note_path.write_text(note_body, encoding="utf-8")
    return str(note_path)


def clear_history() -> None:
    """Clear the chat history stored in Streamlit session state."""

    st.session_state.setdefault("messages", [])
    st.session_state["messages"] = []
    st.session_state["last_answer"] = None
    st.session_state["last_citations"] = []
    st.session_state["last_question"] = None


def _slugify(text: str) -> str:
    base = text.strip().lower()
    base = re.sub(r"[^a-z0-9\-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    return base or "note"
