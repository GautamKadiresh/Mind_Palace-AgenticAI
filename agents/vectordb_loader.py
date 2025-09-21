# Import libraries for document processing and configuration
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join

# CONSTANTS
from constants import DB_LOCATION, EMBEDDING_MODEL, SRC_FILES_LOCATION, VECTORS_CHUNK_SIZE, VECTOR_OVERLAP_SIZE


def document_reader(source_file_path):
    onlyfiles = [join(source_file_path, f) for f in listdir(source_file_path) if isfile(join(source_file_path, f))]
    all_documents = []
    for file in onlyfiles:
        print(f"Loading document: {file}")
        ext = Path(file).suffix.lower()  # Extract file extension for format check
        if file == join(source_file_path, "README.md"):
            continue  # Skip readme file
        elif ext == ".txt":
            # Text files
            loader = TextLoader(file)
        elif ext == ".pdf":
            # PDF files
            loader = PyPDFLoader(file)
        else:
            print(f"Unsupported file format: {ext}")
            continue

        document = loader.load()
        all_documents.extend(document)
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def vectordb_loader():
    all_documents = document_reader(SRC_FILES_LOCATION)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=VECTORS_CHUNK_SIZE, chunk_overlap=VECTOR_OVERLAP_SIZE)
    chunked_documents = text_splitter.split_documents(all_documents)
    ids = [str(i) for i in range(len(chunked_documents))]

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(collection_name="mind_palace", persist_directory=DB_LOCATION, embedding_function=embeddings)

    vectorstore_ids = vector_store.get()["ids"]
    if len(vectorstore_ids) != 0:
        vector_store.delete(ids=vectorstore_ids)

    print(f"Adding knowledge vector database...")
    vector_store.add_documents(documents=chunked_documents, ids=ids)
    print(f"Finished adding {len(ids)} vectors to the vector database...")
