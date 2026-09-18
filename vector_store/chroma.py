import os 
from langchain_community.vectorstores import Chroma


PERSIST_ROOT = "chroma_db"

def vector_store_exists(collection_name: str) -> bool:
     """Checks cache on my disk for the uploaded pdf """

     persist_directory = os.path.join(PERSIST_ROOT, collection_name)
     return os.path.isdir(persist_directory) and bool (os.listdir(persist_directory))


def create_vector_store(chunks, embedding_model, collection_name : str) -> Chroma:

     persist_directory = os.path.join(PERSIST_ROOT, collection_name)
     vector_store = Chroma.from_documents(
          documents= chunks,
          embedding= embedding_model,
          persist_directory= persist_directory,
          collection_name= collection_name,
     )

     vector_store.persist
     return vector_store

def load_vector_store(embedding_model, collection_name: str) -> Chroma:
     """loads cache database skips OCR"""

     persist_directory = os.path.join(PERSIST_ROOT, collection_name)

     return Chroma(
          persist_directory= persist_directory,
          embedding_function= embedding_model,
          collection_name= collection_name,

     )