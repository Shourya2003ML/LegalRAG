import os
from retriever.retriever import BasicRAGRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompts.prompts import NAIVE_RAG_PROMPT, QUERY_REWRITE_PROMPT
from configs.config import GROQ_MODEL, RAG_TYPE, EMBEDDING_MODEL
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from guardrails.guardrail import LLMGuardrail


load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
api_key = SecretStr(groq_key) if groq_key else None

class BasicRAGPipeline:
    def __init__(
        self, 
        data_dir, 
        groq_model = GROQ_MODEL,
        rag_type = RAG_TYPE,
        chroma_dir = None
    ):
        self.rag_type = rag_type
        self.retriever = BasicRAGRetriever(data_dir, rag_type=rag_type, chroma_dir = chroma_dir)
        self.llm = ChatGroq(temperature = 0.2, model = groq_model, api_key = api_key, stop_sequences=[])
        self.guardrail = LLMGuardrail()

    def _rewrite_query(self, query, chat_history):
        """
        Rewrites the query into a standlone questions using chat history
        """
        
        #when no history write nothing
        if not chat_history or len(chat_history) < 2:
            return query
        
        #Formatting query rewritting
        history_text = ""
        for turn in chat_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        prompt = QUERY_REWRITE_PROMPT.format(
            history = history_text,
            question = query
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        rewritten = str(response.content).strip() if hasattr(response, "content") else query

        print(f"Original query: {query}")
        print(f"Rewritten query: {rewritten}")

        return rewritten

    def answer(self, query, chat_history = None, top_k = 3, use_reranking = True, use_query_rewriting = True):

        #guardrail check
        is_safe, reason = self.guardrail.check_input(query)
        if not is_safe:
            return {
                "answer" : "This query cannot be processed. Please ask genuine legal question related to the uploaded documents.",
                "sources": [],
                "rag_type": self.rag_type,
                "retrieval_technique": "Blocked by input guardrail",
                "embedding_model": EMBEDDING_MODEL,
                "top_k": top_k,
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "rewritten_query": query,
                "rewritting_used": False,
                "reranking_used": False,
                "blocked" : True,
                "block_reason": reason
            }

        #calling rewrite query if enabled
        if use_query_rewriting:
            rewritten_query = self._rewrite_query(query, chat_history)
            rewriting_used = rewritten_query != query
        else:
            rewritten_query = query
            rewriting_used = False

        

        #Retrieval

        results = self.retriever.retrieve(rewritten_query, top_k = top_k, use_reranking=use_reranking)
        contexts = "\n\n".join([r["content"] for r in results])
        sources = list(set(r["source"] for r in results))
        
        messages = [] 
        system_prompt = NAIVE_RAG_PROMPT.format(context = contexts, question = query)
        
        #using systemmessage wrapper
        messages.append(SystemMessage(content = system_prompt))

        if chat_history:
            for turn in chat_history:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content = turn["content"]))
                elif turn["role"] == "assistant":
                    messages.append(AIMessage(content = turn["content"]))

        #Add current question
        messages.append(HumanMessage(content = query))

        response = self.llm.invoke(messages)  
        content =  str(response.content) if hasattr(response, 'content') else response

        #output guardrail
        output_safe, output_reason = self.guardrail.check_output(content)
        if not output_safe:
            content = "The generated response did not meet quality standard. Please rephrase your question or try again."
        
        return{
            "answer" : content,
            "sources" : sources,
            "rag_type" : self.rag_type,
            "retrieval_technique" : "Hybrid (BM25 + Semantic Search + Reranking)",
            "embedding_model" : EMBEDDING_MODEL,
            "top_k" : top_k,
            "reranker_model" : "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rewritten_query" : rewritten_query,
            "rewriting_used" : rewriting_used,
            "reranking_used" : use_reranking,
            "blocked": False,
            "blocked_reason": "",
            "contexts": [r["content"] for r in results],
        }

    def get_pipeline_info(self):
        """Getting info about current pipeline and collection"""
        return self.retriever.get_collection_info()