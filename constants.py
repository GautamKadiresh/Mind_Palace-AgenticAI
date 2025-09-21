#CHAT ASSISTANT CONSTANTS
LLM_ASSISTANT="llama3.2"
SYSTEM_PROMPT = '''
You are a helpful AI assistant. 
You also have access to a vector database that contains information on various topics added by the user.
Your task is to assist users by providing accurate and relevant information based on their queries.

To answer, you may or may not need to use the retrieved info from the vector database.
If the retrieved info was used then mention the source from the metadata field.

If you do not know the answer, respond with "I don't know".
'''

USER_PROMPT = '''
User Query: 
{user_query}
Information from Vector DB: 
{retrieved_info}
'''


#VECTOR DB CONSTANTS
DB_LOCATION = ".\chroma_vector_db"
EMBEDDING_MODEL = "mxbai-embed-large"
MAX_RESULTS = 1
SRC_FILES_LOCATION = ".\knowledge_source_files"
VECTORS_CHUNK_SIZE = 1000
VECTOR_OVERLAP_SIZE = 200