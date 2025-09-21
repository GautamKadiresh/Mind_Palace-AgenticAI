from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# CONSTANTS
from constants import DB_LOCATION
from constants import EMBEDDING_MODEL
from constants import MAX_RESULTS


embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(collection_name="mind_palace", persist_directory=DB_LOCATION, embedding_function=embeddings)


retriever = vector_store.as_retriever(search_kwargs={"k": MAX_RESULTS})
