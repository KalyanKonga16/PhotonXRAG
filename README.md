# PhotonX Copilot - RAG-Powered Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit** and powered by **Groq's Llama 3.3 LLM**. This project combines hybrid retrieval (BM25 + dense embeddings), reciprocal rank fusion, and cross-encoder reranking to deliver accurate, context-aware answers from your document library.

## Project Overview

PhotonX Copilot is a chat-first application that:

- **Ingests documents** (.docx files) from the `source_docs/` directory
- **Creates vector embeddings** using BAAI/bge-base-en-v1.5 sentence transformers
- **Performs hybrid retrieval** combining lexical (BM25) and semantic (dense embedding) search
- **Reranks results** using a cross-encoder for improved relevance
- **Generates responses** via Groq's Llama 3.3-70b model with full document context
- **Stores vectors** in a persistent Chroma database

### Key Features

- 🚀 **Hybrid Search**: Combines BM25 and dense embeddings via reciprocal rank fusion
- 🎯 **Smart Reranking**: Cross-encoder reranking ensures top results are truly relevant
- ⚡ **Fast LLM**: Powered by Groq's fast inference service (free tier available)
- 💬 **Streamlit UI**: Clean, modern chat interface with dark theme
- 📚 **Document Processing**: Automatic ingestion and chunking of .docx files
- 💾 **Persistent Storage**: Vector database cached locally for instant responses

## Setup Instructions

### Prerequisites

- Python 3.9+
- Windows, macOS, or Linux
- Git (optional, for version control)

### 1. Clone or Download the Project

```bash
cd /path/to/PhotonXRAG
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies included:**
- `streamlit` - Web UI framework
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `rank_bm25` - BM25 retrieval
- `groq` - Groq API client
- `langchain-text-splitters` - Document chunking
- `python-docx` - Word document parsing
- `python-dotenv` - Environment variable management

### 4. Set Up Environment Variables

Create or configure `.streamlit/secrets.toml` in the project root:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
HF_TOKEN = "your_huggingface_token_here"  # Optional, for model access
```

**How to get keys:**
- **GROQ_API_KEY**: Sign up at [console.groq.com](https://console.groq.com) (free tier available)
- **HF_TOKEN**: Optional; get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if needed for model downloads

### 5. Prepare Your Documents

Place your `.docx` files in the `source_docs/` directory:

```
source_docs/
  ├── document1.docx
  ├── document2.docx
  └── ...
```

### 6. Ingest Documents

Run the ingestion script to process documents and build the vector database:

```bash
python ingest.py
```

This will:
- Parse all `.docx` files in `source_docs/`
- Split documents into chunks
- Generate embeddings using sentence-transformers
- Store vectors in `chroma_db/`
- Save metadata to `ingest_metadata.json`

## Running the Project Locally

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will start at `http://localhost:8501` by default.

### Access the Chat Interface

- Open your browser to `http://localhost:8501`
- Type your questions in the chat input
- The RAG engine will retrieve relevant documents and generate accurate responses

### Development Notes

- **Hot reload disabled**: The `config.toml` file has `fileWatcherType = "none"` to prevent latency issues with large dependencies. Restart `streamlit run app.py` manually after code changes.
- **First run**: Initial embedding generation will take time as models are downloaded (~500MB total).

## Project Structure

```
PhotonXRAG/
├── app.py                    # Streamlit web interface
├── rag_engine.py             # RAG retrieval & generation logic
├── ingest.py                 # Document ingestion pipeline
├── debug_retrieve.py         # Debugging tool for retrieval
├── requirements.txt          # Python dependencies
├── config.toml              # Streamlit configuration
├── .streamlit/
│   └── secrets.toml         # API keys (git ignored)
├── source_docs/             # Input documents (.docx files)
├── chroma_db/               # Vector database (git ignored)
└── README.md                # This file
```

## Required Environment Variables

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `GROQ_API_KEY` | String | API key for Groq's Llama models | ✅ Yes |
| `HF_TOKEN` | String | Hugging Face token for model downloads | ❌ Optional |

**Note**: These should be set in `.streamlit/secrets.toml` and are automatically git-ignored.

## RAG Pipeline Configuration

Edit `rag_engine.py` to adjust retrieval parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `DENSE_TOP_K` | 20 | Dense embedding search results |
| `BM25_TOP_K` | 20 | BM25 lexical search results |
| `RRF_K` | 60 | Reciprocal rank fusion constant |
| `RERANK_CANDIDATE_CAP` | 60 | Candidates for cross-encoder reranking |
| `FINAL_TOP_N` | 6 | Chunks sent to LLM for generation |

### Models Used

- **Embedding Model**: `BAAI/bge-base-en-v1.5` (from HuggingFace)
- **Reranker Model**: `BAAI/bge-reranker-base` (from HuggingFace)
- **LLM**: `llama-3.3-70b-versatile` (from Groq)

## Troubleshooting

### App won't start
- Ensure virtual environment is activated
- Verify all dependencies: `pip install -r requirements.txt`
- Check `GROQ_API_KEY` is set in `.streamlit/secrets.toml`

### Slow initial startup
- First run downloads embedding models (~500MB). Subsequent runs are instant.
- Disable Streamlit's file watcher (already done in `config.toml`)

### No documents found
- Ensure `.docx` files are in `source_docs/` directory
- Run `python ingest.py` to process documents
- Check `ingest_metadata.json` for ingestion logs

### Embedding errors
- Verify internet connection (models are downloaded from HuggingFace)
- Try clearing model cache: `rm -rf ~/.cache/huggingface/`
- Rerun `python ingest.py`

## API Endpoints

This is a Streamlit application (not REST API-based). However, the core RAG logic in `rag_engine.py` can be imported and used in other applications (FastAPI, Flask, etc.):

```python
from rag_engine import load_resources, ask

# Load resources once
load_resources()

# Ask a question
response = ask("What is PhotonX?")
print(response)
```

## Performance Notes

- **Retrieval Speed**: ~200-500ms per query (hybrid search + reranking)
- **Generation Speed**: ~2-5 seconds for typical responses (Groq free tier)
- **Memory Usage**: ~2-3GB when models are loaded (embeddings + reranker + LLM context)
- **Vector DB**: Grows with document size (~100KB per DOCX page)

## Future Enhancements

- [ ] FastAPI REST endpoints for integration
- [ ] Support for additional document formats (PDF, TXT, markdown)
- [ ] Batch document processing
- [ ] Query expansion and multi-turn conversation history
- [ ] Custom prompt templates
- [ ] Model switching and fine-tuning
- [ ] Monitoring and analytics dashboard

## Additional Dependencies

### System Dependencies

None required for basic operation. All ML models are downloaded automatically from HuggingFace and Groq.

### Optional: If Building from Scratch

```bash
# Install with development tools
pip install -r requirements.txt
pip install pytest black flake8  # Testing & formatting (optional)
```

## License

This project is provided as-is for research and personal use.

## Support & Issues

- **Groq API Issues**: Check [Groq Console](https://console.groq.com/docs/api-documentation)
- **Model Issues**: Visit [HuggingFace Model Cards](https://huggingface.co/BAAI)
- **Streamlit Help**: See [Streamlit Docs](https://docs.streamlit.io)

---

**Last Updated**: July 2026  
**Python Version**: 3.9+  
**Status**: Active Development
