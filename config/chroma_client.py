from langchain_chroma import Chroma
import chromadb
from config.constants import CHROMA_DB_PATH
from config.llm_client import embeddings

# ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_chroma_collection(name: str) -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=name,
    )
