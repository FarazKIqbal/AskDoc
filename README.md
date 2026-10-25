<div align="center">

# HR Policy Assistant

**A RAG-based document Q&A assistant — upload a PDF handbook, ask questions, get grounded answers with page citations.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-ACCCA0?style=flat-square)](https://docs.trychroma.com)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6.1-009688?style=flat-square)](https://www.sentence-transformers.org)
[![pytest](https://img.shields.io/badge/pytest-8.3-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)

</div>

---

## What It Is

HR Policy Assistant is a chatbot that helps employees understand workplace policies by answering questions in plain language. Upload any PDF handbook, ask a question, and get an answer grounded in the document with page citations showing exactly where the information came from.

The entire pipeline is written by hand — no frameworks like LangChain or LlamaIndex. This means every step (PDF parsing, chunking, embedding, retrieval, generation) is visible and understandable.

## How It Works

```
Upload PDF → Extract Text → Chunk (≈500 words, 50-word overlap)
    ↓
Embed Chunks → Store in ChromaDB → Query Top-4 Similar Chunks
    ↓
Build Prompt → Call LLM → Return Answer with Page Citations
```

The system detects when a question isn't covered by the document and responds with "I couldn't find that in the document" instead of guessing.

## Quick Start

```bash
# Install dependencies
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your API key

# Run the app
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and start asking questions.

## Running Evaluations

```bash
python eval/run_eval.py --provider openai --top-k 4
```

The `eval/` directory contains 15 questions with expected answers against the sample PDF. The script reports overall accuracy and lists any failed questions.

## Project Structure

```
Simple RAG/
├── app.py                # Streamlit UI
├── rag/
│   ├── loader.py         # PDF → text with page numbers
│   ├── chunker.py        # Text → overlapping chunks
│   ├── store.py          # Embeddings + ChromaDB storage
│   ├── generator.py      # Prompt building + LLM calls
│   └── pipeline.py       # Ingest and ask functions
├── eval/                 # Evaluation questions and runner
├── data/sample.pdf       # Sample HR policy document
└── tests/                # Unit tests
```

## What I Built

• **Built a RAG-based assistant that answers employee questions about workplace policies in plain language, with answers cited by page number.**

• **Embedded policy documents with Sentence Transformers and stored them in ChromaDB for retrieval, then used the GenAI API to generate the answers.**

• **Created a Streamlit interface for employees to ask questions.**

## Technologies Used

- **UI**: Streamlit
- **Vector Store**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, free
- **LLM**: Anthropic Claude or OpenAI (configurable via environment)
- **PDF Parsing**: pypdf
- **Testing**: pytest

## License

MIT

---

Built as a portfolio project to demonstrate understanding of RAG systems.
