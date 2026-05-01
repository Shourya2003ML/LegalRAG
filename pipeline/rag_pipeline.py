import os
from retriever.retriever import BasicRAGRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompts.prompts import NAIVE_RAG_PROMPT
from configs.config import GROQ_MODEL, RAG_TYPE, EMBEDDING_MODEL

from pydantic import SecretStr
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
api_key = SecretStr(groq_key) if groq_key else None

class BasicRAGPipeline:
    def __init__(
        self, 
        data_dir, 
        groq_model = GROQ_MODEL,
        rag_type = RAG_TYPE
    ):
        self.rag_type = rag_type
        self.retriever = BasicRAGRetriever(data_dir, rag_type=rag_type)
        self.llm = ChatGroq(temperature = 0.2, model = groq_model, api_key = api_key)

    def answer(self, query, top_k = 3):
        results = self.retriever.retrieve(query, top_k = top_k)
        contexts = "\n\n".join([r["content"] for r in results])
        sources = list(set(r["source"] for r in results)) 
        prompt = NAIVE_RAG_PROMPT.format(context = contexts, question = query)
        response = self.llm.invoke(prompt)  
        content =  response.content if hasattr(response, 'content') else response
        return{
            "answer" : content,
            "sources" : sources,
            "rag_type" : self.rag_type,
            "retrieval_technique" : "Cosine Similarity Search",
            "embedding_model" : EMBEDDING_MODEL,
            "top_k" : top_k
        }

    def get_pipeline_info(self):
        """Getting info about current pipeline and collection"""
        return self.retriever.get_collection_info()