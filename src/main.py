import os
import shutil
from document_loaders.pdf import load_pdf
from document_loaders.ocr import pdf_load_ocr
from text_splitters.splitters import split_documents
from embeddings.embeddings import get_embedding_model
from vector_store.chroma import ( create_vector_store)
from retriever.retriever import get_mmr_retriever
from gen_ai.gen import generate_answer
from coqui.tts import speak, play_audio

CHROMA_DB_DIR = "chroma_db"

def ingest_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    documents = pdf_load_ocr(pdf_path)
    print(f"Loaded {len(documents)} pages")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(chunks, embedding_model)
    print(f"Vector store created for the current PDF\n")
    return vector_store

def ask(question , retriever):

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])
    answer = generate_answer(context, question)

    audio_path = speak(answer)
    if audio_path:
        play_audio(audio_path)

    return answer

def main():

    pdf_path = input("Enter the path for your pdf: ").strip().strip('"')

    vector_store = ingest_pdf(pdf_path)

    retriever = get_mmr_retriever(vector_store)
    print("Ask Questions from the PDF")
    print("Type 'exit' to quit the program\n")

    while True:

        question = input("Ask a question: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("Sayonara!")
            break

        answer = ask(question, retriever)

        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()  

