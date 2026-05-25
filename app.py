import streamlit as st
import os
import shutil
import atexit
from dotenv import load_dotenv

load_dotenv()

#directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "source_data", "naive_rag")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
RAG_TYPE = "naive-rag"
MAX_FILE_MB = 10

# Clear any leftover data from previous sessions on fresh app load
def startup_cleanup():
    try:
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                fpath = os.path.join(DATA_DIR, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
    except Exception as e:
        print(f"Startup cleanup error: {e}")

# Only run once per server start, not on every Streamlit rerun
if "startup_done" not in st.session_state:
    startup_cleanup()
    st.session_state.startup_done = True


#title of the page
st.set_page_config(page_title = "LegalRAG", layout = "centered")
st.title("LegalRAG")
st.caption("Upload Legal PDFs, index them, and then ask questions.")

#session
defaults = {
    "messages": [],
    "pipeline": None,
    "indexed": False,
    "session_files": [],
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

#when session ends
def cleanup_session():
    try:
        for f in st.session_state.get("session_files", []):
            fpath = os.path.join(DATA_DIR, f)
            if os.path.exists(fpath):
                os.remove(fpath)
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
    except Exception as e:
        print(f"Cleanup error: {e}")

atexit.register(cleanup_session)

#Main sidebar
with st.sidebar:
    st.header("Documents")
    st.caption(f"Max file size: {MAX_FILE_MB} MB per file")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type = ["pdf"], accept_multiple_files = True
    )

    st.divider()
    st.header("Settings")

    top_k = st.slider("Chunks to retrieve (Top K)", min_value = 1, max_value = 10, value = 3)

    use_reranking = st.toggle(
        "Enable reranking",
        value = True,
        help = "Reranking helps improving the answers by using Cross Encoder Model but increases latency." \
        "Recommended if not getting good results"
    )

    use_query_rewriting = st.toggle(
        "Enable query rewriting",
        value = True,
        help = "Rewriting improves follow-up questions before retrieval. Improves responses for long conversations with follow ups"
    )

    st.divider()
    index_btn = st.button("Index Documents", use_container_width = True, type="primary")

    if st.session_state.pipeline is not None:
        st.divider()
        info = st.session_state.pipeline.get_pipeline_info()
        st.success("Pipeline Ready!!")
        st.write(f"**Collection:** `{info['collection_name']}`")
        st.write(f"**Chunks indexed:** `{info['document_count']}`")
        st.write(f"**RAG type:** `{info['rag_type']}`")

    if st.session_state.indexed:
        st.divider()
        if st.button("Clear Session Data", use_container_width = True):
            cleanup_session()
            for key, val in defaults.items():
                st.session_state[key] = val
            st.rerun()

#Indexing of the documents
if index_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first")
    elif not os.getenv("GROQ_API_KEY"):
        st.sidebar.error("GROQ_API_KEY not found.")
    else:
        oversized = [f.name for f in uploaded_files if f.size > MAX_FILE_MB * 1024 * 1024]
        if oversized:
            st.sidebar.error(
                f"These files exceed the {MAX_FILE_MB} MB limit: \n"
                + "\n".join([f"- {n}" for n in oversized])
            )
        else:
            startup_cleanup()
            
            os.makedirs(DATA_DIR, exist_ok = True)
            saved = []
            for f in uploaded_files:
                dest = os.path.join(DATA_DIR, f.name)
                with open(dest, "wb") as out:
                    out.write(f.read())
                saved.append(f.name)
            st.session_state.session_files = saved

            with st.spinner(f"Indexing {len(saved)} file(s)..."):
                try:
                    from pipeline.rag_pipeline import BasicRAGPipeline
                    pipeline = BasicRAGPipeline(data_dir = DATA_DIR, rag_type = RAG_TYPE)
                    pipeline.retriever.index_pdfs()
                    st.session_state.pipeline = pipeline
                    st.session_state.indexed = True
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Indexing failed: {e}")

def show_retrieval_info(msg):
    sources        = msg.get("sources", [])
    reranking_used = msg.get("reranking_used", False)
    rewriting_used = msg.get("rewriting_used", False)
    rewritten_q    = msg.get("rewritten_query", "")
    top_k_used     = msg.get("top_k", "-")
    rag_type       = msg.get("rag_type", RAG_TYPE)
 
    with st.expander("Sources and retrieval info"):
        st.write("**Documents used:**")
        if sources:
            for src in sources:
                st.write(f"- `{src}`")
        else:
            st.write("- No sources recorded")
 
        st.divider()
        st.write("**Retrieval details:**")
        st.write(f"- **RAG pipeline:** `{rag_type}`")
        st.write(f"- **Embedding model:** `all-MiniLM-L6-V2`")
        st.write(f"- **Chunks retrieved:** `{top_k_used}`")
 
        technique = "BM25 + Semantic Search"
        if reranking_used:
            technique += " + Reranking"
        st.write(f"- **Retrieval technique:** `{technique}`")
 
        if reranking_used:
            st.write(f"- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`")
 
        if rewriting_used and rewritten_q:
            st.write(f"- **Query rewriting:** on")
            st.write(f"- **Rewritten query:** _{rewritten_q}_")
        else:
            st.write(f"- **Query rewriting:** off")
 
#chat area
if not st.session_state.indexed:
    st.info("Upload your legal PDFs in the sidebar and click Index Documents to get started.")
else:
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                show_retrieval_info(msg)
 
    # Chat input
    if query := st.chat_input("Ask a legal question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
 
        with st.chat_message("assistant"):
            pipeline = st.session_state.pipeline
 
            # Show query rewriting status before retrieval
            if use_query_rewriting and len(st.session_state.messages) > 1:
                st.caption("Rewriting query using conversation history...")
 
            with st.spinner("Retrieving and generating answer..."):
                try:
                    if pipeline is None:
                        raise ValueError("Pipeline not initialized.")
 
                    history = st.session_state.messages[-6:]
                    result = pipeline.answer(
                        query,
                        chat_history=history,
                        top_k=top_k,
                        use_reranking=use_reranking,
                        use_query_rewriting=use_query_rewriting,
                    )
                    answer          = result["answer"]
                    sources         = result["sources"]
                    rag_type        = result["rag_type"]
                    top_k_used      = result["top_k"]
                    rewritten_query = result.get("rewritten_query", "")
                    rewriting_used  = result.get("rewriting_used", False)
                    reranking_used  = result.get("rerankin  g_used", False)
 
                except Exception as e:
                    answer          = f"Error: {e}"
                    sources         = []
                    rag_type        = RAG_TYPE
                    top_k_used      = top_k
                    rewritten_query = ""
                    rewriting_used  = False
                    reranking_used  = False
 
            st.markdown(answer)
 
            # Build msg dict for display
            assistant_msg = {
                "role":           "assistant",
                "content":        answer,
                "sources":        sources,
                "rag_type":       rag_type,
                "top_k":          top_k_used,
                "reranking_used": reranking_used,
                "rewriting_used": rewriting_used,
                "rewritten_query": rewritten_query,
            }
            show_retrieval_info(assistant_msg)
 
        st.session_state.messages.append(assistant_msg)
 
    # Clear chat
    if st.session_state.messages:
        st.divider()
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()
