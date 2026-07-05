# RAG-Based Document Question Answering System

## Overview
A Retrieval-Augmented Generation (RAG) system that answers questions
based on custom PDF documents using Cohere and Pinecone.

## Tech Stack
- **Cohere** — Embeddings + Answer Generation
- **Pinecone** — Vector Database
- **Streamlit** — Web Interface
- **LangChain** — Text Chunking

## Project Structure
- `app.py` — Main Streamlit web app
- `vectorstore.py` — PDF processing, embeddings, Pinecone storage
- `chatbot.py` — Answer generation using Cohere LLM
- `requirements.txt` — All dependencies

## How to Run

### 1. Activate virtual environment
source .venv/bin/activate

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the app
python -m streamlit run app.py

### 4. Open in browser
http://localhost:8501

## How It Works
1. Upload any PDF document
2. System extracts and chunks the text
3. Chunks converted to embeddings using Cohere
4. Embeddings stored in Pinecone vector database
5. User asks a question
6. Relevant chunks are retrieved from Pinecone
7. Cohere LLM generates a grounded answer

## API Keys Required
- Cohere → https://cohere.com (free tier available)
- Pinecone → https://pinecone.io (free starter plan available)

## Dataset Reference
- HuggingFace: vectara/open_ragbench
- GitHub Reference: VivekChauhan05/RAG_Document_Question_Answering