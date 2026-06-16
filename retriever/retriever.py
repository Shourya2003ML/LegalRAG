from langchain_chroma import Chroma
from utils.pdf_utils import load_pdfs_from_folder
from configs.config import ( TOP_K, RAG_TYPE, BM25_TOP_K, VECTOR_TOP_K, RERANK_TOP_K, RERANKER_MODEL)
from configs.retriever_config import get_retriever_config
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import time
import os

load_dotenv()

class BasicRAGRetriever:
    def __init__(self, data_dir, rag_type = RAG_TYPE, chroma_dir = None):
        self.data_dir = data_dir
        self.rag_type = rag_type
        self.config = get_retriever_config(rag_type)

        self.embedding = self.config["embedding"]
        self.text_splitter = self.config["text_splitter"]
        self.collection_name = self.config["collection_name"]
        self.persist_directory = chroma_dir if chroma_dir else self.config["persist_directory"]
        self.vectorstore = None

        #Reranking and BM 25 for hybrid retrieval
        self.bm25 = None
        self.bm25_chunks = []
        self.bm25_metadata = []
        self.reranker = CrossEncoder(RERANKER_MODEL)


    def index_pdfs(self):
        print(f"Indexing PDFs for collection: {self.collection_name}")
        time.sleep(0.5)
        os.makedirs(self.persist_directory, exist_ok = True)
        pdf_texts = load_pdfs_from_folder(self.data_dir)
        docs = []   
        for item in pdf_texts:
            chunks = self.text_splitter.create_documents(
                [item["text"]],
                metadatas = [{"source": item["filename"]}]
            )
            docs.extend(chunks)

        #ChromaDB Index
        self.vectorstore = Chroma.from_documents(
            docs,
            self.embedding,
            persist_directory = self.persist_directory,
            collection_name = self.collection_name
        )
        print(f"Successfully indexed {len(docs)} documents in collection : {self.collection_name}")

        #BM25 index
        self.bm25_chunks = [doc.page_content for doc in docs]
        self.bm25_metadata = [doc.metadata for doc in docs]
        tokenized = [chunk.lower().split() for chunk in self.bm25_chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Successfully indexed {len(docs)} chunks.")

    def _vector_search(self, query, top_k):
        """
        Semantic search via ChromaDB
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory = self.persist_directory,
                embedding_function = self.embedding,
                collection_name = self.collection_name
            )
        docs = self.vectorstore.similarity_search(query, k = top_k)
        return [
            {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in docs
        ]

    def _bm25_search(self, query, top_k):
        """
        Keyword search via BM25
        """
        if self.bm25 is None:
            #rebuilding from ChromaDB chunks
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key =lambda i : scores[i], reverse = True)[:top_k]
        return [
            {
                "content" : self.bm25_chunks[i],
                "source" : self.bm25_metadata[i].get("source", "Unknown")
            }
            for i in top_indices
        ]

    def _rebuild_bm25(self):
        """
        Rebuilding the BM25 index from ChromaDB when restarting app
        """
        print("Rebuilding BM25 index from ChromaDB")
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory = self.persist_directory,
                embedding_function = self.embedding,
                collection_name = self.collection_name
            )
        #fetching all documents in the ChromaDB
        all_docs = self.vectorstore._collection.get()
        self.bm25_chunks = all_docs["documents"]
        self.bm25_metadata = all_docs["metadatas"]
        tokenized = [chunk.lower().split() for chunk in self.bm25_chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 rebuilt with {len(self.bm25_chunks)} chunks.")
    
    def _rerank(self, query, candidates, top_k):
        """
        Rerank merged candidates using CrossEncoder
        """
        if not candidates:
            return []
        
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key = lambda x: x[0], reverse = True)
        return [c for _, c in ranked[:top_k]]

    def retrieve(self, query, top_k=TOP_K, use_reranking=True):
        vector_results = self._vector_search(query, top_k=VECTOR_TOP_K)
        bm25_results   = self._bm25_search(query, top_k=BM25_TOP_K)

        seen, merged = set(), []
        for result in vector_results + bm25_results:
            if result["content"] not in seen:
                seen.add(result["content"])
                merged.append(result)

        if use_reranking:
            return self._rerank(query, merged, top_k=RERANK_TOP_K)
        return merged[:top_k]
    
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