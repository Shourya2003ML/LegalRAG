# type of rag
RAG_TYPE = "naive-rag"

ALLOWED_COLLECTIONS = "naive_rag_collection"

# where the documents are stored
DATA_DIR = "data/source_data/naive_rag"

#Initializing Chroma DB vectorStore
#for storing locally using persistent storage
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-V2"

#Fetching Top 5 relevant Chunks
TOP_K = 5

#LLM Model
GROQ_MODEL = "openai/gpt-oss-20b"
