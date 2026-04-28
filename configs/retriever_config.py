from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.config import PERSIST_DIR, EMBEDDING_MODEL
from utils.chroma_utils import get_collection_name_for_rag_type

def get_retriever_config(rag_type: str):
    return {
        "embedding": HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        "text_splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        "collection_name": get_collection_name_for_rag_type(rag_type),
        "persist_directory": PERSIST_DIR,
        "vectorstore": None
    }