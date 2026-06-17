# LegalRAG

LegalRAG is a Retrieval-Augmented Generation (RAG) application for legal document analysis. Upload any legal PDF — contracts, agreements, acts — and ask questions about it in plain English. The system retrieves the most relevant chunks from the document using a hybrid retrieval approach and uses an LLM to generate accurate, context-grounded answers.

---

## Use Cases

- Querying specific sections or clauses from legal acts and statutes
- Asking questions about contracts and agreements without reading the entire document
- Understanding legal terminology within the context of an uploaded document
- Comparing provisions across multiple uploaded legal documents
- Legal research and document summarization for students and practitioners

---

## System Architecture

```
User Query
    |
    v
Input Guardrail (LLM-based jailbreak and injection detection)
    |
    v
Query Rewriting (resolves pronouns and vague references using chat history)
    |
    v
Hybrid Retriever
    |-- BM25 (keyword search)          --+
    |-- ChromaDB (semantic search)     --+--> Merge and Deduplicate
                                               |
                                               v
                                         CrossEncoder Reranker
                                               |
                                               v
                                         Top K Chunks
    |
    v
LLM (Groq) with retrieved context + short term memory
    |
    v
Output Guardrail (response quality check)
    |
    v
Answer + Source Citation + Retrieval Info
```

---

## Hybrid Retrieval

LegalRAG uses a three-stage hybrid retrieval pipeline instead of simple vector similarity search:

**Stage 1 - BM25 (keyword search)**
BM25 is a classical keyword-based retrieval algorithm. It is particularly effective for legal text because legal documents use precise, domain-specific terminology. When a user asks about "Section 12" or "Article 3(b)", BM25 finds chunks containing those exact terms reliably.

**Stage 2 - Semantic Search (ChromaDB)**
Semantic search uses the `all-MiniLM-L6-V2` embedding model to find chunks that are conceptually similar to the query even when the exact words differ. For example, a query about "director responsibilities" will match chunks discussing "obligations of board members".

**Stage 3 - Reranking (CrossEncoder)**
After merging and deduplicating results from BM25 and semantic search, a CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores all candidates by reading the query and each chunk together. This is more accurate than embedding-based scoring because the model sees the full context of both the query and the chunk simultaneously. Reranking is optional and can be toggled in the sidebar.

---

## Query Rewriting

In multi-turn conversations, follow-up questions often contain unresolved references:

```
User: What is Section 12?
User: What are its penalties?       <- "its" refers to Section 12
User: Can it be appealed?           <- "it" still refers to Section 12
```

Without query rewriting, the retriever receives vague queries like "What are its penalties?" and returns irrelevant chunks because BM25 and semantic search have no awareness of conversation history.

Query rewriting uses the Groq LLM to rewrite the user's latest question into a fully self-contained query before sending it to the retriever:

```
"What are its penalties?"
->  "What are the penalties under Section 12 of the Companies Act?"
```

The rewritten query goes to the retriever. The original question goes to the LLM so the answer feels natural. Query rewriting is optional and can be toggled in the sidebar.

---

## Guardrails

LegalRAG includes an LLM-based guardrail layer to make the system production-safe.

**Input Guardrail**
Every user query passes through an input guardrail before reaching the retrieval pipeline. The guardrail classifies the query as SAFE or UNSAFE and blocks:
- Jailbreak attempts (e.g. "ignore previous instructions", "act as", "pretend you are")
- Prompt injection attacks (e.g. "your new instructions are", "forget everything")
- Requests for harmful or illegal content

Legitimate queries — including off-topic ones — pass through. Off-topic queries are handled naturally by the RAG prompt which instructs the LLM to state when information is not present in the document.

**Output Guardrail**
After the LLM generates a response, an output guardrail checks whether the response is grounded in the retrieved context or contains unsupported claims. Responses that fail the output check are replaced with a message asking the user to rephrase.

Note: Both guardrails add one LLM call each per query. On Groq this adds minimal latency (under 1 second each).

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Shourya2003ML/LegalRAG.git
cd LegalRAG
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your GROQ API key
Export it in your terminal session:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com).

---

## Run Locally

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Usage:**
1. Upload one or more legal PDFs in the sidebar
2. Click **Index Documents** to embed and store them in ChromaDB
3. Ask questions in the chat — the app will retrieve relevant chunks and generate an answer
4. Expand **Sources and Retrieval Info** under each answer to see which document was used and what retrieval technique was applied

---

## Run with Docker

### Prerequisites
```bash
docker --version
docker compose version
```

If not installed:
```bash
sudo apt-get update
sudo apt install docker.io
sudo apt install docker-compose-plugin
```

On Linux, add your user to the docker group to avoid permission errors:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Build and Run
```bash
export GROQ_API_KEY=your_groq_api_key_here
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Day to Day Docker Commands

| Situation | Command |
|---|---|
| First time / after code changes | `docker compose up --build` |
| Start without rebuilding | `docker compose up` |
| Run in background | `docker compose up -d` |
| View logs when running in background | `docker compose logs -f` |
| Stop the app | `docker compose down` |
| Full reset | `docker compose down --rmi all -v` |

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (`openai/gpt-oss-20b`) via `langchain-groq` |
| Embeddings | `all-MiniLM-L6-V2` via `langchain-huggingface` |
| Keyword Retrieval | BM25 via `rank-bm25` |
| Vector Store | ChromaDB (local persistent storage) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| PDF Parsing | PyMuPDF (`fitz`) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| RAG Pipeline | Hybrid RAG (BM25 + Semantic Search + Reranking) |
| Guardrails | LLM-based input and output safety checks |
| Query Rewriting | LLM-based standalone query generation |
| Containerization | Docker and Docker Compose |

---

## Project Structure

```
LegalRAG/
├── app.py                        # Streamlit app
├── pipeline/
│   └── rag_pipeline.py           # RAG pipeline (retrieval + LLM + guardrails)
├── retriever/
│   └── retriever.py              # Hybrid retriever (BM25 + ChromaDB + reranker)
├── guardrails/
│   ├── __init__.py
│   └── guardrail.py              # Input and output LLM guardrails
├── configs/
│   ├── config.py                 # Global config (model, paths, etc.)
│   └── retriever_config.py       # Embeddings and splitter config
├── prompts/
│   └── prompts.py                # Prompt templates (RAG, rewriting, guardrails)
├── utils/
│   ├── pdf_utils.py              # PDF loading with PyMuPDF
│   └── chroma_utils.py           # ChromaDB utility functions
├── eval/
│   ├── eval.py                   # RAGAS evaluation script
│   └── test_data.json            # Ground truth Q&A pairs for evaluation
├── data/
│   └── source_data/
│       └── naive_rag/            # Place your PDFs here
├── chroma_db/                    # Auto-created vector store (session-specific)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

---

## Notes

- Never commit your API key. Add these to `.gitignore`:
  ```
  .env
  chroma_db/
  __pycache__/
  .venv/
  ```
- Session data (uploaded PDFs and ChromaDB index) is cleared automatically when a new session starts. Documents are not stored permanently by design.
- After changing any PDF or updating the pipeline, delete `chroma_db/` and re-index:
  ```bash
  rm -rf chroma_db/
  ```
- To run the RAGAS evaluation:
  ```bash
  python eval/eval.py
  ```