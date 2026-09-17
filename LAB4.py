# fix for the sqlite3 error on streamlit cloud - chroma needs a newer version.
# this swaps in pysqlite3 (from requirements.txt) and has to run BEFORE chromadb is imported.
# the try/except means it still runs fine in codespaces where pysqlite3 isn't installed.
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

#  settings 
PDF_FOLDER = Path(__file__).parent / "pdfs"      # the 7 course pdfs live in here
COLLECTION_NAME = "Lab4Collection"               # name of my chromadb collection (Part A step 2)
EMBED_MODEL = "text-embedding-3-small"           # openai embeddings model for the vectors
CHAT_MODEL = "gpt-5-mini"                         # same llm i used in HW3
BUFFER_MESSAGES = 6                              # keep the last 6 messages, like HW3 (3 exchanges)
MAX_CHARS_FOR_EMBEDDING = 15000                 # cut each pdf here so it stays under the token limit

st.title("📚 Lab 4 — Course Info Chatbot (RAG)")

#  api key 
# reading the key from secrets, same as HW3 (secrets.toml locally, Secrets settings on the cloud)
if "OPENAI_API_KEY" not in st.secrets:
    st.error("No OPENAI_API_KEY found in secrets. Add it to .streamlit/secrets.toml "
             "(and to your app's Secrets on Streamlit Cloud).")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


#  build the vector database (Part A step 2) 
def build_lab4_vectordb(api_key: str):
    # this reads all 7 pdfs, turns them into text, and stores them in chromadb.
    # chroma calls openai to make the embeddings for me on add() and query().

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    chroma_client = chromadb.Client()  # in-memory, lives for this session
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    # grab the pdfs from the pdfs/ folder (and check the main folder too just in case)
    pdf_paths = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_paths:
        pdf_paths = sorted(Path(__file__).parent.glob("*.pdf"))
    if not pdf_paths:
        st.error("No PDFs found. Put the 7 PDF files in a 'pdfs/' folder next to LAB4.py.")
        st.stop()

    documents, ids, metadatas = [], [], []
    for pdf_path in pdf_paths:
        # read every page of the pdf and join the text together
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        text = text.strip()
        if not text:
            continue  # skip a pdf if no text comes out (e.g. a scanned image)

        documents.append(text[:MAX_CHARS_FOR_EMBEDDING])
        ids.append(pdf_path.name)                       # using the filename as the key
        metadatas.append({"filename": pdf_path.name})   # keep the filename in metadata too

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # hold on to the client so python doesn't garbage-collect my in-memory data
    st.session_state._lab4_chroma_client = chroma_client
    return collection


# only build the db once and save it in session_state - this way i'm not paying to
# re-embed the pdfs every time the app reruns (Part A step 2)
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building the course knowledge base (one-time embedding)…"):
        st.session_state.Lab4_VectorDB = build_lab4_vectordb(openai_api_key)

collection = st.session_state.Lab4_VectorDB


#  course info chatbot with RAG (Part B) 
st.subheader("Ask a question about the course")

# start the chat history in session_state so it survives reruns
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the course."}
    ]

# show the conversation so far
for msg in st.session_state.lab4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your question about the course"):
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG bit: search the vector db for the 3 most relevant syllabi for this question
    results = collection.query(query_texts=[prompt], n_results=3)
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]

    # glue the retrieved text together to feed into the prompt
    context = ""
    for doc_id, doc_text in zip(retrieved_ids, retrieved_docs):
        context += f"\n--- From {doc_id} ---\n{doc_text[:3000]}\n"

    # prompt engineering: hand the model the retrieved docs and tell it to say
    # when it's actually using them (that's the "be clear about RAG" part, step 5)
    system_prompt = (
        "You are a helpful course-information assistant. "
        "Use the COURSE DOCUMENTS below to answer the student's question. "
        "If your answer uses information from those documents, clearly say so — "
        "for example, begin with 'Based on the course materials...' and mention "
        "which file(s) it came from. If the documents do not contain the answer, "
        "say that you are answering from general knowledge instead.\n\n"
        f"COURSE DOCUMENTS:\n{context}"
    )

    # keep only the last few messages (short memory like HW3), then always put the
    # system prompt with the fresh context first so it never gets dropped
    buffered = st.session_state.lab4_messages[-BUFFER_MESSAGES:]
    api_messages = [{"role": "system", "content": system_prompt}] + buffered

    with st.chat_message("assistant"):
        # stream the answer back like in lab 1 and HW3
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=api_messages,
            stream=True,
        )
        answer = st.write_stream(stream)

    # save the reply into memory
    st.session_state.lab4_messages.append({"role": "assistant", "content": answer})