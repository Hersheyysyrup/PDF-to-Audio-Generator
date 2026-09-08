import re
import streamlit as st
import os
from gen_ai.gen import generate_answer
from coqui.tts import speak
from document_loaders.ocr import pdf_load_ocr
from text_splitters.splitters import split_documents
from embeddings.embeddings import get_embedding_model
from vector_store.chroma import create_vector_store
from retriever.retriever import get_mmr_retriever

import time
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.time()
    yield
    st.session_state.setdefault("timings", []).append((label, time.time() - start))


st.set_page_config(page_title="PDF to Audio Generator",
                    layout="wide",
                    initial_sidebar_state = "expanded")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .stChatMessage { border-radius: 12px; padding: 0.5rem 1rem; }
    div[data-testid="stChatMessage"] { margin-bottom: 0.75rem; }
    .doc-status {
        padding: 0.6rem 0.8rem; border-radius: 8px;
        background-color: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.4);
        font-size: 0.85rem; margin-top: 0.5rem;
    }
    .stButton button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

#removing aestrick and other sound pronounciation

def clear_for_tts(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Sidebar
with st.sidebar:
    st.markdown("PDF to Audio Generator")
    st.caption("Upload a PDF, ask questions, and get audio responses.")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    load_col, clear_col = st.columns([2, 1])
    with load_col:
        load_clicked = st.button("Load PDF", use_container_width=True, type="primary",
                                  disabled=uploaded_file is None)
    with clear_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.messages = []
            st.session_state.doc_name = None
            st.rerun()

    if uploaded_file and load_clicked:
        progress = st.progress(0, text="Reading PDF...")
        temp_path = os.path.join("temp_uploads", uploaded_file.name)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress.progress(30, text="Running OCR...")
        with timer("OCR/Load"):
            documents = pdf_load_ocr(temp_path)

        progress.progress(55, text="Splitting into chunks...")
        with timer("Split"):
            chunks = split_documents(documents)

        progress.progress(75, text="Embedding...")
        with timer("Embed model load"):
            embedding_model = get_embedding_model()
        with timer("Vector store build"):
            vector_store = create_vector_store(chunks, embedding_model)

        progress.progress(95, text="Building retriever...")
        st.session_state.retriever = get_mmr_retriever(vector_store)
        st.session_state.messages = []
        st.session_state.doc_name = uploaded_file.name
        st.session_state.chunk_count = len(chunks)

        progress.progress(100, text="Done")
        st.rerun()

    if st.session_state.doc_name:
        st.markdown(
            f'<div class="doc-status"> <b>{st.session_state.doc_name}</b><br>'
            f'{st.session_state.chunk_count} chunks indexed</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No document loaded yet.")

    if st.session_state.get("timings"):
        with st.expander("⏱️ Timing breakdown"):
            for label, secs in st.session_state.timings[-8:]:
                st.write(f"{label}: {secs:.2f}s")

    st.divider()
    st.caption(" Ask anything about the uploaded PDF below.")


# Main chat interface
st.title("Good to see you!")

if not st.session_state.doc_name:
    st.markdown("Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("audio") and os.path.exists(msg["audio"]):
                st.audio(msg["audio"])
            if msg.get("pages"):
                st.caption(" Sources: " + ", ".join(f"Page {p}" for p in msg["pages"]))

    if question := st.chat_input("Ask a question about your PDF..."):
        if st.session_state.retriever is None:
            st.warning("Upload and load a PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    with timer("Retrieval"):
                        docs = st.session_state.retriever.invoke(question)
                    context = "\n".join(doc.page_content for doc in docs)
                    with timer("LLM answer"):
                        answer = generate_answer(context, question)
                with st.spinner("Generating audio..."):
                    with timer("TTS"):
                        audio_path = speak(clear_for_tts(answer))
                pages = sorted(set(doc.metadata.get("page") for doc in docs if doc.metadata.get("page")))

                st.write(answer)
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path)
                if pages:
                    st.caption(" Sources: " + ", ".join(f"Page {p}" for p in pages))

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "audio": audio_path, "pages": pages
            })