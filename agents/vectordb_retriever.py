from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# CONSTANTS
from constants import VECTOR_DB_CONFIG


embeddings = OllamaEmbeddings(model=VECTOR_DB_CONFIG["EMBEDDING_MODEL"])
vector_store = Chroma(collection_name="mind_palace", persist_directory=VECTOR_DB_CONFIG["DB_LOCATION"], embedding_function=embeddings)


retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_DB_CONFIG["MAX_RESULTS"]})