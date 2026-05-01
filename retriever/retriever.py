from langchain_chroma import Chroma
from utils.pdf_utils import load_pdfs_from_folder
from configs.config import TOP_K, RAG_TYPE
from configs.retriever_config import get_retriever_config
from dotenv import load_dotenv

load_dotenv()

class BasicRAGRetriever:
    def __init__(self, data_dir, rag_type = RAG_TYPE):
        self.data_dir = data_dir
        self.rag_type = rag_type
        self.config = get_retriever_config(rag_type)

        self.embedding = self.config["embedding"]
        self.text_splitter = self.config["text_splitter"]
        self.collection_name = self.config["collection_name"]
        self.persist_directory = self.config["persist_directory"]
        self.vectorstore = self.config["vectorstore"]

    def index_pdfs(self):
        print(f"Indexing PDFs for collection: {self.collection_name}")
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []   
        for item in pdf_texts:
            chunks = self.text_splitter.create_documents(
                [item["text"]],
                metadatas = [{"sources": item["filename"]}]
            )
            docs.extend(chunks)
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory = self.persist_directory,
            collection_name = self.collection_name
        )
        print(f"Successfully indexed {len(docs)} documents in collection : {self.collection_name}")

    def retrieve(self, query, top_k = TOP_K):
        if self.vectorstore is None:
            print(f"Loading existing vector store for collection: {self.collection_name}")
            self.vectorstore = Chroma(
                persist_directory = self.persist_directory,
                embedding_function = self.embedding,
                collection_name = self.collection_name
            )
        docs = self.vectorstore.similarity_search(query, k = top_k)

        #checking what source is returned by chromadb
        print(docs[0].metadata)
        
        results = []
        for doc in docs:
            results.append({
                "content":doc.page_content,
                "source" : doc.metadata.get("sources", )
            })
        return results
    
    def get_collection_info(self):
        """Current collection status"""
        if self.vectorstore is None:
            self.retrieve("test", top_k = 1)
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "rag_type" : self.rag_type
            }
        except Exception as e:
            return {
                "collection_name" : self.collection_name,
                "document_count" : 0,
                "rag_type" : self.rag_type,
                "error": str(e)
            }
        
if __name__ == "__main__":
    retriever = BasicRAGRetriever(data_dir = "data/naive_rag", rag_type = "naive-rag")
    retriever.index_pdfs()
    print(retriever.get_collection_info())