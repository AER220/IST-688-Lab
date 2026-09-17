# =============================================================================
# sqlite3 shim — MUST come before importing chromadb.
# Streamlit Community Cloud ships an old system sqlite3 that ChromaDB rejects.
# This swaps in the newer 'pysqlite3-binary' wheel (installed via requirements.txt).
# The try/except lets the file still run locally / in Codespaces without it.
# =============================================================================
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from pathlib import Path

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------- Config --------------------------------------
PDF_FOLDER = Path(__file__).parent / "pdfs"      # put the 7 PDF files in this folder
COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"           # OpenAI embeddings model
CHAT_MODEL = "gpt-5-mini"                         # same model your HW3 uses
BUFFER_MESSAGES = 6                               # short memory, like HW3 (3 exchanges)
MAX_CHARS_FOR_EMBEDDING = 15000                  # keep each doc safely under the token limit

st.title("📚 Lab 4 — Course Info Chatbot (RAG)")

# ------------------------------- API key -------------------------------------
# Same approach as HW3: read the key from .streamlit/secrets.toml (locally) or
# from the app's Secrets settings on Streamlit Cloud.
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# ------------------- Build the ChromaDB collection (once) --------------------
def build_lab4_vectordb(api_key: str):
    """Read the 7 PDFs, embed them with OpenAI, and store them in a ChromaDB
    collection called 'Lab4Collection'. Returns the collection."""

    # ChromaDB will call OpenAI to create embeddings for us on add() and query().
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    chroma_client = chromadb.Client()  # in-memory client (lives for this session)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    # Look in the pdfs/ folder first; if empty, fall back to the repo root.
    pdf_paths = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_paths:
        pdf_paths = sorted(Path(__file__).parent.glob("*.pdf"))
    if not pdf_paths:
        st.error("No PDFs found. Put the 7 PDF files in a 'pdfs/' folder next to Lab4.py.")
        st.stop()

    documents, ids, metadatas = [], [], []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        text = text.strip()
        if not text:
            continue  # skip PDFs with no extractable text (e.g., scanned images)

        documents.append(text[:MAX_CHARS_FOR_EMBEDDING])
        ids.append(pdf_path.name)                       # filename is the key
        metadatas.append({"filename": pdf_path.name})   # metadata

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # Keep the client referenced so the in-memory data isn't garbage-collected.
    st.session_state._lab4_chroma_client = chroma_client
    return collection


# Only build the DB if we haven't already — this avoids paying to re-embed
# the PDFs every time the script reruns.
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building the course knowledge base (one-time embedding)…"):
        st.session_state.Lab4_VectorDB = build_lab4_vectordb(openai_api_key)

collection = st.session_state.Lab4_VectorDB


# ============================================================================
# PART A — retrieval test.
# Use this to confirm the vector DB returns sensible files, then DELETE this
# whole block for your Part B final submission (lab step 4).
# ============================================================================
with st.expander("🔎 Part A — test the vector DB (delete before final submit)"):
    test_query = st.text_input("Test search string", value="Generative AI")
    if test_query:
        results = collection.query(query_texts=[test_query], n_results=3)
        st.write("Top 3 matching documents:")
        for i, doc_id in enumerate(results["ids"][0], start=1):
            st.write(f"{i}. {doc_id}")
# ============================================================================


# ============================================================================
# PART B — course info chatbot with RAG.
# ============================================================================
st.subheader("Ask a question about the course")

if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the course."}
    ]

# Show the conversation so far.
for msg in st.session_state.lab4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your question about the course"):
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- RAG: fetch the 3 most relevant course documents for this question ----
    results = collection.query(query_texts=[prompt], n_results=3)
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]

    context = ""
    for doc_id, doc_text in zip(retrieved_ids, retrieved_docs):
        context += f"\n--- From {doc_id} ---\n{doc_text[:3000]}\n"

    # Prompt engineering: give the model the retrieved text and tell it to be
    # explicit about when it is using that RAG knowledge.
    system_prompt = (
        "You are a helpful course-information assistant. "
        "Use the COURSE DOCUMENTS below to answer the student's question. "
        "If your answer uses information from those documents, clearly say so — "
        "for example, begin with 'Based on the course materials...' and mention "
        "which file(s) it came from. If the documents do not contain the answer, "
        "say that you are answering from general knowledge instead.\n\n"
        f"COURSE DOCUMENTS:\n{context}"
    )

    # Keep only the last few messages (short memory, like HW3), then always put
    # the system prompt (with the freshly retrieved context) first.
    buffered = st.session_state.lab4_messages[-BUFFER_MESSAGES:]
    api_messages = [{"role": "system", "content": system_prompt}] + buffered

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=api_messages,
            stream=True,
        )
        answer = st.write_stream(stream)

    st.session_state.lab4_messages.append({"role": "assistant", "content": answer})