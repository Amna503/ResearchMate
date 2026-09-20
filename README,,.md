# ResearchMate

**Your AI Research & Learning Mate**
*Learn from your research. Ask. Explore. Understand.*

## Overview

ResearchMate is an intermediate-level **Retrieval-Augmented Generation (RAG)** application built
with Streamlit. It helps students and researchers understand, explore, and learn from their own
uploaded research papers — instead of getting generic, ungrounded answers from a chatbot.

This is a genuine RAG system: every answer is generated from a small set of retrieved passages
pulled from your uploaded PDFs, not from the model's general knowledge, and not by dumping the
whole PDF into the prompt.

## Problem

Students and beginners often struggle with:

- Understanding long, dense research papers
- Finding specific information buried inside a paper
- Comparing multiple papers against each other
- Identifying methodology, findings, limitations, and research gaps
- Understanding technical terminology
- Knowing exactly where an answer came from

## Solution

ResearchMate lets you upload one or more PDF papers, then chat with them. Every response is
grounded in retrieved evidence from the papers themselves, and shows you exactly which paper,
page, and passage the answer is based on. If the papers don't contain enough information, the
app tells you so instead of guessing.

## Features

- **Multi-PDF upload** with per-file processing status, page counts, and chunk counts
- **Metadata-aware chunking** — every chunk keeps its source file and page number
- **Semantic embeddings** via Sentence Transformers (configurable model)
- **FAISS vector search** with cosine similarity and a relevance threshold
- **Configurable Top-K, chunk size, and chunk overlap** from the sidebar
- **8 Learning Modes**: General Q&A, Explain Simply, Deep Dive, Key Findings, Methodology,
  Limitations, Research Gap, and Compare Papers (for multi-document comparison)
- **Source citations** on every answer (paper name + page number)
- **"View retrieved evidence"** expander showing the exact passages used
- **Chat history** persisted in session state for natural follow-up questions
- **Graceful error handling** for missing API keys, empty PDFs, no-context questions, etc.
- **No hardcoded secrets** — reads the Groq API key from Streamlit secrets or environment
  variables only

## RAG Architecture

ResearchMate strictly follows the RAG pipeline — it never sends an entire PDF to the LLM:

```
PDFs
  ↓
Text Extraction (PyMuPDF, per page)
  ↓
Cleaning + Chunking (character-based, with overlap, page-level metadata)
  ↓
Embeddings (Sentence Transformers, e.g. all-MiniLM-L6-v2)
  ↓
FAISS Vector Store (cosine similarity via normalized inner product)
  ↓
Semantic Retrieval (Top-K most relevant chunks, above a similarity threshold)
  ↓
Relevant Context (only the retrieved chunks are passed forward)
  ↓
LLM (Groq) — instructed to answer only from retrieved context
  ↓
Evidence-Based Answer + Sources + Retrieved Passages shown in the UI
```

## How It Works

1. You upload one or more PDF research papers.
2. Each page's text is extracted with PyMuPDF, cleaned, and split into overlapping chunks.
   Each chunk stores `{source, page, chunk_id}` metadata.
3. Chunks are embedded with a Sentence Transformers model and added to a FAISS index
   (`IndexFlatIP` over L2-normalized vectors, which is equivalent to cosine similarity).
4. When you ask a question, the question is embedded the same way and FAISS returns the
   Top-K most similar chunks, filtered by a minimum similarity threshold.
5. The retrieved chunks (with their source/page labels) are inserted into the LLM prompt as
   context. The system prompt instructs the model (via Groq) to answer only from that context,
   to say explicitly when the evidence is insufficient, to cite sources, and to distinguish
   direct evidence from interpretation.
6. The UI displays the answer, a list of sources, and an expandable "View retrieved evidence"
   section with the actual passages used — full RAG transparency.

## Tech Stack

| Layer            | Technology                              |
|-------------------|------------------------------------------|
| UI / App          | Streamlit                                |
| PDF Extraction     | PyMuPDF (`fitz`)                         |
| Embeddings         | Sentence Transformers (`all-MiniLM-L6-v2`)|
| Vector Store       | FAISS (`faiss-cpu`)                      |
| LLM                | Groq API (GPT-OSS 120B / 20B models)     |
| Language           | Python 3.10+                             |

No agent framework is used — this project intentionally implements a clean, understandable
RAG pipeline before introducing any agentic behavior.

## Project Structure

```
researchmate/
│
├── app.py                          # Full Streamlit RAG application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Excludes secrets, caches, venvs
├── .env.example                    # Template for local env vars (no real key)
└── .streamlit/
    └── secrets.toml.example        # Template for Streamlit secrets (no real key)
```

`app.py` is organized into clear sections: config, PDF processing, embeddings, FAISS vector
store, retrieval, LLM calls, session state, and Streamlit UI rendering — so it stays readable
for someone learning how RAG works, while still being a single deployable file.

## Local Setup

1. **Clone or download the project**, then move into the folder:
   ```bash
   cd researchmate
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a free Groq API key** from [console.groq.com/keys](https://console.groq.com/keys).

## Environment Variables / Streamlit Secrets

Never hardcode your API key. Choose ONE of the following:

**Option A — Streamlit secrets (recommended for local + Streamlit Cloud):**
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key
```

**Option B — Environment variable:**
```bash
cp .env.example .env
# edit .env and paste your real key, then export it, e.g.:
export GROQ_API_KEY=your_key_here
```

Both `.env` and `.streamlit/secrets.toml` are already excluded via `.gitignore`.

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to
   your repository and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_real_groq_api_key"
   ```
4. Deploy. Streamlit Cloud will install `requirements.txt` automatically.

No local file paths or local-only dependencies are used, so the app runs the same way in the
cloud as it does locally. Uploaded documents and the FAISS index live entirely in Streamlit's
session state for the duration of a user's session — nothing is persisted to disk on the server.

## Example Questions

- "What methodology does this paper use?"
- "What are the key findings of this study?"
- "What limitations does the author mention?"
- "Explain the results section in simple terms."
- "Compare the datasets used in these two papers."
- "Is there a research gap identified in this work?"

## Limitations

- Retrieval quality depends on chunk size/overlap and the embedding model — very short or
  very technical papers may need tuned settings in the sidebar.
- Character-based chunking is simple by design (for clarity/learning); it does not use
  sentence- or section-aware splitting.
- Scanned/image-only PDFs with no extractable text will not produce usable chunks (no OCR yet).
- All state is session-based — re-uploading is required after a browser refresh or new session.
- This version does not use AI agents or an agent framework by design; it is a focused RAG
  system meant to be understood end-to-end first.

## Future Improvements

- Optional OCR fallback for scanned PDFs
- Sentence/section-aware chunking
- Persistent vector store across sessions (e.g. saved FAISS index per user)
- Reranking retrieved chunks with a cross-encoder for higher precision
- Exportable chat/answer history (PDF or Markdown)
- An agentic layer on top of this RAG core (explicitly out of scope for this version)
