"""Streamlit UI for the RAG document Q&A app.

- Upload a PDF
- Ask questions in a chat box
- See answers grounded in the document with page citations
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv

import streamlit as st

# Load environment variables from .env (API keys, RAG_PROVIDER, etc.)
load_dotenv()

from rag.pipeline import DEFAULT_CHROMA_DIR, ingest, ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋", layout="wide")

st.title("📋 HR Policy Assistant")
st.caption(
    "Upload your company HR policy handbook and ask questions in plain "
    "language. Answers are drawn from the document and cited by page."
)

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "Model provider",
        options=["openai", "claude", "local"],
        format_func=lambda x: {
            "openai": "OpenAI-compatible (API)",
            "claude": "Claude (API)",
            "local": "Local (free, offline)",
        }[x],
        index=0,
    )
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)
    st.divider()
    st.caption(
        "Tip: The local model is a small open LLM. It may hallucinate on "
        "questions the document doesn't cover. The out-of-scope guard tries "
        "to catch those."
    )

# File uploader
uploaded = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    help="The PDF is processed once per session. Re-upload to change documents.",
)

# Persistent state across re-runs
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_pdf" not in st.session_state:
    st.session_state.ingested_pdf = None

# Ingest the PDF when it changes
if uploaded is not None:
    pdf_name = uploaded.name
    if st.session_state.ingested_pdf != pdf_name:
        with st.spinner("Reading PDF and building index..."):
            # Save to a temp file so the pipeline can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            try:
                ingest(tmp_path)
                st.session_state.ingested_pdf = pdf_name
                st.session_state.messages = []  # reset chat on new document
                st.success(f"Indexed {pdf_name}")
            except Exception as e:
                logger.exception("Ingestion failed")
                # A scanned/image PDF has no extractable text — say so
                # plainly instead of dumping a ValueError at the user.
                if "no extractable text" in str(e).lower():
                    st.error(
                        "This PDF has no extractable text — it may be a scanned "
                        "image or an empty document. Try a different file."
                    )
                else:
                    st.error(f"Failed to ingest PDF: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

# Chat input
if st.session_state.ingested_pdf:
    prompt = st.chat_input("Ask a question about the document...")
    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve and generate
        with st.spinner("Thinking..."):
            try:
                answer, chunks = ask(
                    prompt,
                    provider=provider,
                    top_k=top_k,
                )
            except Exception as e:
                logger.exception("Query failed")
                answer = f"Error: {e}"
                chunks = []

        # Append assistant message with source info
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [{"page": c.page_number, "text": c.text} for c in chunks],
            }
        )

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            # Show sources in an expander
            if msg.get("sources"):
                with st.expander("📖 View source passages"):
                    for src in msg["sources"]:
                        st.markdown(f"**Page {src['page']}**")
                        st.write(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                        st.divider()
else:
    st.info("👈 Upload a PDF to start asking questions.")

# Footer
st.divider()
st.caption(
    "Built with sentence-transformers (all-MiniLM-L6-v2), ChromaDB, "
    "and an LLM of your choice (OpenAI-compatible, Anthropic, or local Qwen2.5)."
)