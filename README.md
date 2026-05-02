# ⚖️ LegalRAG

LegalRAG is a Retrieval-Augmented Generation (RAG) application for legal document analysis. Upload any legal PDF — contracts, agreements, acts — and ask questions about it in plain English. The system retrieves the most relevant chunks from the document and uses an LLM to generate accurate, context-grounded answers.

---

## 🚀 Setup

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
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com).

---

## ▶️ Run Locally

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

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (`openai/gpt-oss-20b`) via `langchain-groq` |
| Embeddings | `all-MiniLM-L6-V2` via `langchain-huggingface` |
| Vector Store | ChromaDB (local persistent storage) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| RAG Pipeline | Naive RAG (cosine similarity search) |
| Config & Env | `python-dotenv` |

---

## 📁 Project Structure

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
├── .env                          # Your GROQ_API_KEY (never commit this)
├── .dockerignore
├── Dockerfile
└── requirements.txt
```

---

## ⚠️ Notes

- Never commit your `.env` file. Add it to `.gitignore`:
  ```
  .env
  chroma_db/
  __pycache__/
  ```
- After changing any PDF or updating the pipeline, delete `chroma_db/` and re-index:
  ```bash
  rm -rf chroma_db/
  ```
