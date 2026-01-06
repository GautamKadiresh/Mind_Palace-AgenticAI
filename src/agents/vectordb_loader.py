# Import libraries for document processing and configuration
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from pathlib import Path
from os import listdir
from os.path import isfile, join

# Agent configs
from configs import VECTOR_DB_CONFIG

# terminal output formatting constants
from constants import DIM_CYAN, BOLD_RED, BOLD_YELLOW, BOLD_GREEN, RESET_FONT


def document_reader(files_to_load):
    all_documents = []
    for file in files_to_load:
        ext = Path(file).suffix.lower()  # Extract file extension for format check
        if file.endswith("README.md"):
            continue  # Skip readme file
        elif ext == ".txt":
            # Text files
            loader = TextLoader(file)
        elif ext == ".pdf":
            # PDF files
            loader = PyPDFLoader(file)
        elif ext == ".html" or ext == ".htm":
            # html files
            loader = WebBaseLoader(file)
        else:
            print(f"{BOLD_YELLOW}Unsupported file format: {ext}")
            continue

        print(f"{DIM_CYAN}Loading document: {file}")
        try:
            documents = loader.load()
            # Inject last modified timestamp in vector db document metadata  
            last_modified = os.path.getmtime(file)
            for doc in documents:
                doc.metadata["last_modified"] = last_modified
                doc.metadata["source"] = str(Path(file).resolve())
            all_documents.extend(documents)
        except Exception as e:
            print(f"{BOLD_RED}Failed to load document {file}. Error: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def vectordb_loader():    
    source_file_path = os.path.join(os.getcwd(), VECTOR_DB_CONFIG["SRC_FILES_LOCATION"])
    print(f"{DIM_CYAN}Please wait while the database loads files from {source_file_path}\n")
    db_path = os.path.join(os.getcwd(), VECTOR_DB_CONFIG["DB_LOCATION"])
    if not os.path.isdir(source_file_path):
        os.makedirs(source_file_path)

    # 1. Scan Filesystem: Get current files and their last modified timestamps
    current_files = {}
    if os.path.exists(source_file_path):
        for f in listdir(source_file_path):
            full_path = join(source_file_path, f)
            if isfile(full_path) and not f.endswith("README.md"):
                abs_path = str(Path(full_path).resolve())
                current_files[abs_path] = os.path.getmtime(full_path)

    # 2. Scan DB: Get existing sources and their timestamps
    embeddings = OllamaEmbeddings(model=VECTOR_DB_CONFIG["EMBEDDING_MODEL"])
    vector_store = Chroma(collection_name="mind_palace", persist_directory=db_path, embedding_function=embeddings)

    db_data = vector_store.get(include=["metadatas"])
    db_files = {}  # Map source -> set of IDs (to handle chunks)
    db_timestamps = {}  # Map source -> timestamp (to identify later if file is outdated and needs to be reloaded)

    if db_data["ids"]:
        for i, metadata in enumerate(db_data["metadatas"]):
            if not metadata:
                continue

            source = metadata.get("source")
            last_modified = metadata.get("last_modified")

            if source:
                # Normalize source path just in case
                source_path = str(Path(source).resolve())

                if source_path not in db_files:
                    db_files[source_path] = []
                    db_timestamps[source_path] = last_modified

                db_files[source_path].append(db_data["ids"][i])

    # 3. Determine Actions
    files_to_add = []  # New files
    files_to_update = []  # Existing files with different timestamps
    ids_to_delete = []  # IDs of missing or outdated files

    # Check for deletions (in DB but not in FS)
    for source in db_files:
        if source not in current_files:
            print(f"{DIM_CYAN}Detected deleted file: {source}")
            ids_to_delete.extend(db_files[source])

    # Check for additions and updates
    for file_path, fs_timestamp in current_files.items():
        if file_path not in db_files:
            print(f"{DIM_CYAN}Detected new file: {file_path}")
            files_to_add.append(file_path)
        else:
            db_timestamp = db_timestamps.get(file_path)
            # Strict inequality check for update
            if db_timestamp is None or db_timestamp != fs_timestamp:
                print(f"{DIM_CYAN}Detected modified file: {file_path}")
                files_to_update.append(file_path)
                # Mark existing chunks for deletion
                ids_to_delete.extend(db_files[file_path])
            else:
                # File is up to date
                print(f"{DIM_CYAN}Skipping file as already present in DB : {file_path}")

    # 4. Execute Actions

    # Delete missing or outdated chunks
    if ids_to_delete:
        print(f"{DIM_CYAN}Deleting obsolete chunks...")
        # Chroma requires chunks for deletion if list is too large, but for local 50 max cap this is fine usually.
        # However, to be safe lets do it in batches of 5000 if needed, though rarely hit here.
        vector_store.delete(ids=ids_to_delete)

    # Load new content (Additions + Updates)
    files_to_process = files_to_add + files_to_update
    if files_to_process:
        print(f"{DIM_CYAN}Loading new and updated files...")
        new_documents = document_reader(files_to_process)

        if new_documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=VECTOR_DB_CONFIG["VECTORS_CHUNK_SIZE"], chunk_overlap=VECTOR_DB_CONFIG["VECTOR_OVERLAP_SIZE"]
            )
            chunked_documents = text_splitter.split_documents(new_documents)

            if chunked_documents:
                print(f"{DIM_CYAN}\tAdding new vector chunks...")
                vector_store.add_documents(documents=chunked_documents)
    else:
        print(f"{DIM_CYAN}No new or modified files to process.")

    # Handle empty DB case
    if len(vector_store.get()["ids"]) == 0:
        print(f"{BOLD_YELLOW}\nNo files found in {source_file_path}.\n\nPlease add files for LLM to read. But you can continue to ask questions.")
    else:
        print(f"{DIM_CYAN}Done!!{RESET_FONT}")
