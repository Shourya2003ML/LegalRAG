import streamlit as st
import os
from dotenv import load_dotenv
from pipeline.rag_pipeline import BasicRAGPipeline
from configs.config import EMBEDDING_MODEL

load_dotenv()

#Basic Page Initialization 
st.set_page_config(page_title = "LegalRAG", layout = "centered")

st.title("LegalRAG")

st.caption("Upload legal PDFs, index them, then ask questions")

#Session state

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

#Constants

DATA_DIR = "data/source_data/naive_rag"
RAG_TYPE = "naive-rag"

#Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type = ["pdf"], accept_multiple_files = True
    )

    st.divider()
    st.header("Settings")
    top_k = st.slider("Top K chunks to retrieve", min_value = 1, max_value = 10, value = 3)

    st.divider()
    index_btn = st.button("Index Documents", use_container_width = True, type = "primary")

    #Meta Data 
    if st.session_state.pipeline is not None:
        st.divider()
        info = st.session_state.pipeline.get_pipeline_info()
        st.success("Pipeline ready!")
        st.write(f"**Collection:**'{info['collection_name']}'")
        st.write(f"**Chunks indexed**'{info['document_count']}'")
        st.write(f"**RAG Type**'{info['rag_type']}'")

#Indexing Logic

if index_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    elif not os.getenv("GROQ_API_KEY"):
        st.sidebar.error("GROQ_API_KEY not found. Add it to your .env file")
    else:
        os.makedirs(DATA_DIR, exist_ok = True)
        saved = []
        for f in uploaded_files:
            dest = os.path.join(DATA_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            saved.append(f.name)
        with st.spinner(f"Indexing {len(saved)} files - This will take a minute..."):
            try:
                pipeline = BasicRAGPipeline(data_dir = DATA_DIR, rag_type = RAG_TYPE)
                pipeline.retriever.index_pdfs()
                st.session_state.pipeline = pipeline
                st.session_state.indexed = True
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Indexing Failed: \n\n'{e}'")

if not st.session_state.indexed:
    st.info("Upload your legal PDFs in the sidebar and click **Index Documents** to get started")
else:
    #Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources & Retrieval Info"):
                    st.write("**Documents used:**")
                    for src in msg.get("sources", []):
                        st.write(f"- `{src}`")
                    st.divider()
                    st.write("** Retrieval Details:**")
                    st.write(f"- **Technique:** `Cosine Similarity Search`")
                    st.write(f"- **Embedding model:** `all-MiniLM-L6-V2`")
                    st.write(f"- **RAG pipeline:** `{msg.get('rag_type', 'naive-rag')}`")
                    st.write(f"- **Chunks retrieved:** `{msg.get('top_k', '-')}`")
    
    #Chat input
    if query := st.chat_input("Ask a legal question about your documents..."):

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                try:
                    pipeline = st.session_state.pipeline
                    if pipeline is None:
                        answer = "Pipeline not initialized."
                        sources, rag_type, top_k_used = [], "naive-rag", top_k
                    else:
                        result = pipeline.answer(query, top_k=top_k)
                        answer = result["answer"]
                        sources = result["sources"]
                        rag_type = result["rag_type"]
                        top_k_used = result["top_k"]
                except Exception as e:
                    answer = f"Error: `{e}`"
                    sources, rag_type, top_k_used = [], "naive-rag", top_k

            st.markdown(answer)
            if sources:
                with st.expander("Sources & Retrieval Info"):
                    st.write("**Documents used:**")
                    for src in sources:
                        st.write(f"- `{src}`")
                    st.divider()
                    st.write("**Retrieval Details:**")
                    st.write(f"- **Technique:** `Cosine Similarity Search`")
                    st.write(f"- **Embedding model:** `all-MiniLM-L6-V2`")
                    st.write(f"- **RAG pipeline:** `{rag_type}`")
                    st.write(f"- **Chunks retrieved:** `{top_k_used}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "rag_type": rag_type,
            "top_k": top_k_used
        })

        #Type of rag and the sources used are displayed here
        # with st.expander("Sources & Retrieval Info"):
        #     st.write("Documents Used")
        #     for src in sources:
        #         st.write(f"- `{src}`")
        #     st.divider()
        #     st.write("**Retrieval Details:**")
        #     st.write(f"- **Technique:** `Cosine Similarity Search`")
        #     st.write(f"- **Embedding model:** `{EMBEDDING_MODEL}`")
        #     st.write(f"- **RAG pipeline:** `{rag_type}`")
        #     st.write(f"- **Chunks retrieved:** `{top_k}`")

    #Clear Chat
    if st.session_state.messages:
        st.divider()
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()