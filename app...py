"""
ResearchMate — Your AI Research & Learning Mate
A Retrieval-Augmented Generation (RAG) application for exploring research papers.

Pipeline: PDF -> Text Extraction -> Chunking -> Embeddings -> FAISS -> Retrieval -> LLM (Groq)
"""

import os
import re
import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Optional heavy imports are wrapped so the app can still render a friendly
# error message if a dependency failed to install, instead of crashing hard.
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# ===========================================================================
# CONFIG
# ===========================================================================

APP_NAME = "ResearchMate"
APP_TAGLINE = "Learn from your research. Ask. Explore. Understand."

EMBEDDING_MODEL_OPTIONS = {
    "MiniLM-L6-v2 (fast, recommended)": "sentence-transformers/all-MiniLM-L6-v2",
    "MiniLM-L12-v2 (slightly stronger)": "sentence-transformers/all-MiniLM-L12-v2",
}

GROQ_MODEL_OPTIONS = {
    "GPT-OSS 120B (recommended, high quality)": "openai/gpt-oss-120b",
    "GPT-OSS 20B (fast/cheap)": "openai/gpt-oss-20b",
}

DEFAULT_CHUNK_SIZE = 900        # characters
DEFAULT_CHUNK_OVERLAP = 150     # characters
DEFAULT_TOP_K = 4
MIN_SIMILARITY_THRESHOLD = 0.30  # cosine similarity floor (0-1) for "relevant enough"

LEARNING_MODES = {
    "General Q&A": (
        "Answer the user's question directly and clearly using only the retrieved context."
    ),
    "Explain Simply": (
        "Explain the retrieved research content in simple, beginner-friendly language. "
        "Avoid jargon; where a technical term is unavoidable, briefly define it."
    ),
    "Deep Dive": (
        "Give a detailed, technical explanation of the retrieved content, suitable for a "
        "reader familiar with the field. Preserve technical terminology and nuance."
    ),
    "Key Findings": (
        "Extract and list the key findings/results reported in the retrieved content. "
        "Use a short bulleted list. Only include findings explicitly present in the context."
    ),
    "Methodology": (
        "Explain the research methodology, dataset, experimental setup, or approach "
        "described in the retrieved content. If specific details (e.g. dataset name, "
        "model, evaluation metric) are not present, say so explicitly."
    ),
    "Limitations": (
        "Identify limitations explicitly mentioned in the retrieved content. Only state "
        "limitations the paper itself acknowledges. Do not invent limitations."
    ),
    "Research Gap": (
        "Identify research gaps. Clearly separate two categories in your answer: "
        "(1) gaps EXPLICITLY stated by the paper, and (2) gaps that are YOUR interpretation "
        "based on the retrieved content. Never present interpretation as an explicit "
        "statement from the paper."
    ),
    "Compare Papers": (
        "Compare the retrieved content across the different source papers provided in the "
        "context (they are labeled by source filename). Organize your comparison by aspect "
        "(e.g. Methodology, Dataset, Findings, Limitations, Research Gap) where the context "
        "allows. If a paper does not contain information for a given aspect, explicitly say "
        "'not found in the retrieved content for this paper' rather than guessing."
    ),
}

SYSTEM_PROMPT_TEMPLATE = """You are ResearchMate, an AI research companion that helps students and \
researchers understand their own uploaded papers using Retrieval-Augmented Generation (RAG).

STRICT RULES:
1. Answer using ONLY the retrieved context passages provided below. Do not use outside knowledge \
to fabricate facts about the papers.
2. If the retrieved context does not contain enough information to answer confidently, say clearly: \
"The uploaded papers do not provide enough evidence to answer this confidently." Do not guess.
3. Whenever you state something from the papers, mention which paper and page it came from \
(e.g. "According to Paper_A.pdf, page 4, ..."). Use the [Source: filename, page X] labels attached \
to each passage below.
4. Clearly distinguish between (a) information directly supported by the retrieved text, and \
(b) reasonable interpretation or synthesis you are adding. Label interpretation explicitly, e.g. \
"This is not explicitly stated, but based on the context, one could infer...".
5. Do not invent citations, numbers, or claims that are not present in the context.
6. Current learning mode instruction: {mode_instruction}

Respond in clear, well-structured prose (with bullet points where helpful). Keep the tone helpful \
and educational, like a knowledgeable study partner.
"""


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


@dataclass
class DocRecord:
    file_hash: str
    file_name: str
    num_pages: int
    num_chunks: int
    status: str  # "processed" | "failed" | "empty"


# ===========================================================================
# CACHED RESOURCES
# ===========================================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Add it to requirements.txt."
        )
    return SentenceTransformer(model_name)


def get_groq_client() -> Optional["Groq"]:
    if Groq is None:
        return None
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ===========================================================================
# PDF PROCESSING
# ===========================================================================

def clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph structure."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number, cleaned_text) using PyMuPDF."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Add pymupdf to requirements.txt.")
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            raw = page.get_text("text") or ""
            cleaned = clean_text(raw)
            pages.append((i, cleaned))
    return pages


def chunk_page_text(
    text: str, source: str, page: int, chunk_size: int, overlap: int
) -> List[Chunk]:
    """Split a single page's text into overlapping chunks, tagged with metadata."""
    chunks: List[Chunk] = []
    if not text:
        return chunks

    step = max(chunk_size - overlap, 1)
    start = 0
    idx = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            idx += 1
            chunk_id = f"{source}_{page}_{idx}"
            chunks.append(Chunk(text=piece, source=source, page=page, chunk_id=chunk_id))
        if end == text_len:
            break
        start += step
    return chunks


def process_pdf(
    file_name: str, file_bytes: bytes, chunk_size: int, overlap: int
) -> Tuple[DocRecord, List[Chunk]]:
    file_hash = hashlib.md5(file_bytes).hexdigest()
    try:
        pages = extract_pages(file_bytes)
    except Exception as e:
        return DocRecord(file_hash, file_name, 0, 0, f"failed: {e}"), []

    all_chunks: List[Chunk] = []
    for page_num, page_text in pages:
        if page_text:
            all_chunks.extend(
                chunk_page_text(page_text, file_name, page_num, chunk_size, overlap)
            )

    if not all_chunks:
        return DocRecord(file_hash, file_name, len(pages), 0, "empty (no extractable text)"), []

    return DocRecord(file_hash, file_name, len(pages), len(all_chunks), "processed"), all_chunks


# ===========================================================================
# VECTOR STORE (FAISS)
# ===========================================================================

def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed. Add faiss-cpu to requirements.txt.")
    dim = embeddings.shape[1]
    # Normalize so inner product == cosine similarity
    norm_emb = embeddings.astype("float32").copy()
    faiss.normalize_L2(norm_emb)
    index = faiss.IndexFlatIP(dim)
    index.add(norm_emb)
    return index


def embed_texts(model, texts: List[str]) -> np.ndarray:
    return np.array(model.encode(texts, show_progress_bar=False, convert_to_numpy=True))


def retrieve(
    query: str,
    model,
    index,
    chunks: List[Chunk],
    top_k: int,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
) -> List[Tuple[Chunk, float]]:
    if index is None or not chunks:
        return []
    q_emb = embed_texts(model, [query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score >= similarity_threshold:
            results.append((chunks[idx], float(score)))
    return results


# ===========================================================================
# LLM (GROQ)
# ===========================================================================

def build_context_block(retrieved: List[Tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in retrieved:
        blocks.append(
            f"[Source: {chunk.source}, page {chunk.page}, relevance {score:.2f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def call_llm(
    client,
    model_name: str,
    mode_instruction: str,
    question: str,
    context_block: str,
    chat_history: List[Dict[str, str]],
) -> str:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(mode_instruction=mode_instruction)

    user_prompt = (
        f"RETRIEVED CONTEXT FROM UPLOADED PAPERS:\n{context_block}\n\n"
        f"USER QUESTION:\n{question}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    # Include a short window of prior turns for conversational continuity
    for turn in chat_history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )
    return completion.choices[0].message.content


# ===========================================================================
# SESSION STATE INITIALIZATION
# ===========================================================================

def init_session_state():
    defaults = {
        "doc_records": {},       # file_hash -> DocRecord
        "all_chunks": [],        # List[Chunk] across all processed docs
        "embeddings": None,      # np.ndarray of all chunk embeddings
        "faiss_index": None,
        "chat_messages": [],     # list of {"role": ..., "content": ...} for LLM context
        "display_messages": [],  # list of dicts for rendering (includes sources)
        "embedding_model_name": list(EMBEDDING_MODEL_OPTIONS.values())[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def rebuild_index(model):
    """Rebuild embeddings + FAISS index from st.session_state.all_chunks."""
    chunks = st.session_state.all_chunks
    if not chunks:
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        return
    texts = [c.text for c in chunks]
    embeddings = embed_texts(model, texts)
    st.session_state.embeddings = embeddings
    st.session_state.faiss_index = build_faiss_index(embeddings)


# ===========================================================================
# UI HELPERS
# ===========================================================================

def render_header():
    st.markdown(
        f"""
        <div style="padding: 1.2rem 0 0.4rem 0;">
            <h1 style="margin-bottom:0;">🧭 {APP_NAME}</h1>
            <p style="color:#6b7280; font-size:1.05rem; margin-top:0.2rem;">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "A Retrieval-Augmented Generation (RAG) app — answers are grounded in the "
        "research papers you upload, with sources and evidence shown for every response."
    )


def render_pipeline_diagram():
    with st.expander("🔬 How ResearchMate Works (RAG Pipeline)", expanded=False):
        st.markdown(
            """
            ```
            PDFs
              ↓
            Text Extraction (PyMuPDF, per page)
              ↓
            Cleaning + Chunking (with overlap, page-level metadata)
              ↓
            Embeddings (Sentence Transformers)
              ↓
            FAISS Vector Store (cosine similarity)
              ↓
            Semantic Retrieval (Top-K relevant chunks)
              ↓
            Relevant Context (only chunks above the similarity threshold)
              ↓
            LLM (Groq) — answers using retrieved evidence only
              ↓
            Evidence-Based Answer + Sources + Retrieved Passages
            ```
            ResearchMate never sends your entire PDF to the LLM. Every answer is grounded in a
            small set of the most relevant retrieved passages, and you can always inspect exactly
            which passages were used.
            """
        )


def render_sidebar():
    st.sidebar.header("⚙️ RAG Settings")

    embedding_label = st.sidebar.selectbox(
        "Embedding model", list(EMBEDDING_MODEL_OPTIONS.keys()), index=0
    )
    embedding_model_name = EMBEDDING_MODEL_OPTIONS[embedding_label]

    top_k = st.sidebar.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    chunk_size = st.sidebar.slider(
        "Chunk size (characters)", min_value=300, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk overlap (characters)", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=25
    )

    st.sidebar.markdown("---")
    groq_label = st.sidebar.selectbox("Groq model", list(GROQ_MODEL_OPTIONS.keys()), index=0)
    groq_model_name = GROQ_MODEL_OPTIONS[groq_label]

    st.sidebar.markdown("---")
    st.sidebar.header("📚 Document Library")
    if not st.session_state.doc_records:
        st.sidebar.caption("No papers uploaded yet.")
    else:
        for rec in st.session_state.doc_records.values():
            status_icon = "✅" if rec.status == "processed" else "⚠️"
            st.sidebar.markdown(
                f"**{status_icon} {rec.file_name}**\n\n"
                f"Pages: {rec.num_pages} · Chunks: {rec.num_chunks} · Status: {rec.status}"
            )

    return {
        "embedding_model_name": embedding_model_name,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "groq_model_name": groq_model_name,
    }


def render_upload_section(settings: Dict):
    st.subheader("📤 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    if SentenceTransformer is None or fitz is None or faiss is None:
        st.error(
            "One or more required libraries are missing (pymupdf / sentence-transformers / "
            "faiss-cpu). Please check requirements.txt and your environment."
        )
        return

    new_files_processed = False
    with st.spinner("Loading embedding model..."):
        model = load_embedding_model(settings["embedding_model_name"])

    for f in uploaded_files:
        file_bytes = f.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if file_hash in st.session_state.doc_records:
            continue  # already processed this session — skip silently

        if len(file_bytes) == 0:
            st.warning(f"'{f.name}' appears to be empty and was skipped.")
            continue

        with st.spinner(f"Processing '{f.name}'..."):
            record, chunks = process_pdf(
                f.name, file_bytes, settings["chunk_size"], settings["chunk_overlap"]
            )

        st.session_state.doc_records[file_hash] = record

        if record.status == "processed":
            st.session_state.all_chunks.extend(chunks)
            new_files_processed = True
            st.success(f"'{f.name}' processed: {record.num_pages} pages, {record.num_chunks} chunks.")
        elif record.status.startswith("empty"):
            st.warning(f"'{f.name}' had little or no extractable text (may be a scanned image PDF).")
        else:
            st.error(f"Failed to process '{f.name}': {record.status}")

    if new_files_processed:
        with st.spinner("Building embeddings and FAISS index..."):
            rebuild_index(model)

    processed_count = sum(1 for r in st.session_state.doc_records.values() if r.status == "processed")
    st.caption(
        f"📄 {len(st.session_state.doc_records)} file(s) uploaded this session · "
        f"{processed_count} successfully processed · "
        f"{len(st.session_state.all_chunks)} total chunks indexed."
    )


def render_sources(retrieved: List[Tuple[Chunk, float]]):
    st.markdown("**Sources**")
    seen = []
    for chunk, score in retrieved:
        label = f"{chunk.source} — Page {chunk.page} (relevance {score:.2f})"
        if label not in seen:
            st.markdown(f"- {label}")
            seen.append(label)

    with st.expander("🔎 View retrieved evidence"):
        for i, (chunk, score) in enumerate(retrieved, start=1):
            st.markdown(f"**Passage {i} — {chunk.source}, page {chunk.page} (score {score:.2f})**")
            st.markdown(f"> {chunk.text}")


def render_chat(settings: Dict):
    st.subheader("💬 Chat with Your Papers")

    mode = st.selectbox("Learning Mode", list(LEARNING_MODES.keys()), index=0)

    # Render chat history
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("retrieved"):
                render_sources(msg["retrieved"])

    question = st.chat_input("Ask a question about your uploaded papers...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.display_messages.append({"role": "user", "content": question})

    # --- Guard rails / error handling ---
    if not st.session_state.all_chunks or st.session_state.faiss_index is None:
        answer = (
            "Please upload at least one research paper first — I can only answer questions "
            "based on documents you've uploaded."
        )
        with st.chat_message("assistant"):
            st.warning(answer)
        st.session_state.display_messages.append({"role": "assistant", "content": answer})
        return

    client = get_groq_client()
    if client is None:
        answer = (
            "⚠️ Groq API key not found. Please set `GROQ_API_KEY` in Streamlit secrets or as an "
            "environment variable to enable answers."
        )
        with st.chat_message("assistant"):
            st.error(answer)
        st.session_state.display_messages.append({"role": "assistant", "content": answer})
        return

    model = load_embedding_model(settings["embedding_model_name"])

    with st.spinner("Retrieving relevant passages..."):
        try:
            retrieved = retrieve(
                question,
                model,
                st.session_state.faiss_index,
                st.session_state.all_chunks,
                settings["top_k"],
            )
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            return

    if not retrieved:
        answer = (
            "I couldn't find enough evidence in the uploaded papers to answer this confidently. "
            "Try rephrasing your question, lowering relevance expectations, or uploading a paper "
            "that covers this topic."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.display_messages.append({"role": "assistant", "content": answer})
        return

    context_block = build_context_block(retrieved)
    mode_instruction = LEARNING_MODES[mode]

    if mode == "Compare Papers" and len({c.source for c, _ in retrieved}) < 2:
        st.info(
            "Compare Papers mode works best with multiple uploaded papers. Only one paper's "
            "content was retrieved for this question."
        )

    with st.chat_message("assistant"):
        with st.spinner("Thinking with retrieved evidence..."):
            try:
                answer = call_llm(
                    client,
                    settings["groq_model_name"],
                    mode_instruction,
                    question,
                    context_block,
                    st.session_state.chat_messages,
                )
            except Exception as e:
                answer = f"⚠️ The LLM request failed: {e}"
        st.markdown(answer)
        render_sources(retrieved)

    st.session_state.chat_messages.append({"role": "user", "content": question})
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "retrieved": retrieved}
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
    init_session_state()

    render_header()
    render_pipeline_diagram()

    settings = render_sidebar()

    render_upload_section(settings)
    st.markdown("---")
    render_chat(settings)


if __name__ == "__main__":
    main()
