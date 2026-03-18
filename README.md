# RAG Knowledge Assistant

A retrieval-augmented generation system for asking questions about PDF documents with source-attributed answers.

## What it does

Drop PDF files into a folder, run the indexer, and ask natural language questions. The system retrieves relevant passages and generates answers using an LLM, citing the exact document and page where each piece of information came from.

## How it works

1. **Ingest**: PDFs are parsed and split into overlapping text chunks
2. **Embed**: Chunks are converted to vector embeddings using sentence-transformers
3. **Store**: Embeddings are stored in ChromaDB for fast similarity search
4. **Retrieve**: User queries are embedded and matched against stored chunks
5. **Generate**: Retrieved context is sent to Groq's LLM to generate a cited answer

## Setup

```bash
git clone https://github.com/SanaAraj/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Add PDF files to the `data/` directory.

## Usage

Index your documents:

```bash
python main.py --ingest
```

Ask questions:

```bash
python main.py --query "What is machine learning?"
```

Example output:

```
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

[Source: machine_learning_guide.pdf, Page 1]

--- Sources ---
- machine_learning_guide.pdf, page 1
```

## Tech stack

- Groq API (llama-3.1-8b-instant) for answer generation
- sentence-transformers (all-MiniLM-L6-v2) for local embeddings
- ChromaDB for vector storage
- PyPDF2 for PDF parsing
