# RAG application


A lightweight Retrieval Augmented Generation (RAG) API built with FastAPI. It loads a PDF, chunks it, embeds with HuggingFace embeddings, stores vectors in FAISS, and answers questions using Groq LLMs.

- Entrypoint: `app.api.main:app`
- Default port: `8000`
- Data source: `data/boarding.pdf` (configurable in `app/core/config.py`)

## Requirements
- Python 3.10+
- pip
- (Optional) Docker

## Setup (local)
1) Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set environment variables

This project uses Groq via `langchain_groq`. You need a Groq API key in your environment (loaded by `python-dotenv` if you have a `.env` file).

Create a `.env` file at the repo root with:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

3) Ensure your document is available

By default, the engine loads `data/boarding.pdf`. You can change this path in `app/core/config.py` (constant `PDF_PATH`).

4) Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

5) Try it

```bash
curl --get "http://localhost:8000/ask" --data-urlencode "question=What is this document about?"
```

## Docker
Build and run using the provided Dockerfile.

```bash
docker build -t rag-app:latest -f Docker/Dockerfile .
docker run -d -p 8000:8000 --name rag-app rag-app:latest
```

Check logs:

```bash
docker logs -f rag-app
```

## Project structure

- `app/api/main.py` — FastAPI app exposing `/ask`
- `app/core/config.py` — core settings (PDF path, chunk sizes, embedding model)
- `app/rag/loader.py` — loads text from PDF
- `app/rag/chunker.py` — splits text into chunks
- `app/rag/embeddings.py` — HuggingFace embeddings
- `app/rag/vectorstore.py` — FAISS vector store
- `app/rag/engine.py` — orchestration: load -> chunk -> embed -> search -> LLM
- `data/` — PDF files (input corpus)
- `Docker/Dockerfile` — Docker image definition

## Configuration notes
- Update `app/core/config.py` to tweak chunk size, overlap, embedding model, and `TOP_K` results.
- The embedding model defaults to `sentence-transformers/all-MiniLM-L6-v2` and downloads weights on first run.
- Set `GROQ_API_KEY` in your environment (or `.env`) for the Groq LLM to work.

## Troubleshooting
- Large file push rejected by GitHub: ensure your `myenv/`, `.venv/`, `venv/`, `__pycache__/`, and `.env` are ignored by `.gitignore`.
- SSH permission denied on `git push`: add an SSH key to GitHub or switch your remote to HTTPS.
- PDF not found: verify the path in `app/core/config.py` and that the file exists under `data/`.

## License
Add your license in `LICENSE` (or remove this section).
