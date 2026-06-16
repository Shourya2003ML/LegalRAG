# LegalRAG

LegalRAG is a Retrieval-Augmented Generation (RAG) application for legal document analysis. Upload any legal PDF — contracts, agreements, acts — and ask questions about it in plain English. The system retrieves the most relevant chunks from the document and uses an LLM to generate accurate, context-grounded answers.

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
4. Expand **Sources & Retrieval Info** under each answer to see which document was used

---

## Run with Docker

### Prerequisites
Make sure Docker is installed:
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
Export your GROQ API key first, then start the app:
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
| Full reset (remove image + volumes) | `docker compose down --rmi all -v` |

### Notes on Docker
- Your `chroma_db/` and `data/` folders are mounted as volumes so your index and PDFs persist across container restarts
- After editing any `.py` file, always rebuild with `docker compose up --build`
- After changing PDFs or updating the pipeline, delete the old index before re-indexing:
  ```bash
  rm -rf chroma_db/
  docker compose up --build
  ```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (`openai/gpt-oss-20b`) via `langchain-groq` |
| Embeddings | `all-MiniLM-L6-V2` via `langchain-huggingface` |
| Vector Store | ChromaDB (local persistent storage) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| RAG Pipeline | Naive RAG (cosine similarity search) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
LegalRAG/
├── app.py                        # Streamlit app
├── pipeline/
│   └── rag_pipeline.py           # RAG pipeline (retrieval + LLM)
├── retriever/
│   └── retriever.py              # ChromaDB retriever
├── configs/
│   ├── config.py                 # Global config (model, paths, etc.)
│   └── retriever_config.py       # Embeddings and splitter config
├── prompts/
│   └── prompts.py                # LLM prompt template
├── utils/
│   ├── pdf_utils.py              # PDF loading with PyMuPDF
│   └── chroma_utils.py           # ChromaDB utility functions
├── data/
│   └── source_data/
│       └── naive_rag/            # Place your PDFs here
├── chroma_db/                    # Auto-created vector store
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── .dockerignore                 # Files excluded from Docker image
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
- After changing any PDF or updating the pipeline, delete `chroma_db/` and re-index:
  ```bash
  rm -rf chroma_db/
  ```