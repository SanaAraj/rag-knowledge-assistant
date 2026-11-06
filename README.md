<div align="center">

# RAG Knowledge Assistant

**Ask natural-language questions about your own PDFs and get answers grounded in the documents, with every claim cited back to the exact file and page. When the documents do not contain the answer, it says so instead of guessing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-f55036.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-4b2aad.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/embeddings-local%20(MiniLM)-3776ab.svg)](https://www.sbert.net/)

<img src="docs/demo.gif" alt="Terminal demo: indexing two PDF documents, then asking what supervised learning is and how backpropagation works and getting answers cited to a specific document and page, then asking an out-of-scope question and getting a refusal instead of a hallucination" width="860">

</div>

## What it does

Drop PDFs into a folder, index them once, and ask questions in plain English. The
assistant retrieves the most relevant passages, sends only those to the LLM, and
returns an answer that **cites the document name and page** behind it. Retrieval
runs locally, so your documents never leave your machine for the embedding step,
and the only external call is the final answer generation.

Two properties make it trustworthy rather than a demo toy:

- **Source attribution.** Every answer lists the documents and pages it drew from, so you can verify it.
- **Honest refusal.** If retrieval finds nothing relevant above a similarity threshold, the assistant replies "I don't have enough information" instead of inventing an answer.

## How it works

```
PDFs ─► parse + chunk ─► embed locally (MiniLM) ─► ChromaDB
                                                      │
question ─► embed ─► similarity search ─► top passages ─► Groq LLM ─► cited answer
```

1. **Ingest** - PDFs are parsed with PyPDF2 and split into overlapping word chunks.
2. **Embed** - each chunk is turned into a vector locally with `sentence-transformers` (`all-MiniLM-L6-v2`).
3. **Store** - vectors are kept in a persistent ChromaDB collection (cosine similarity).
4. **Retrieve** - a question is embedded and matched against the store; results below a similarity threshold are discarded.
5. **Generate** - only the retrieved passages are sent to Groq, with a system prompt that forbids answering beyond the provided context and requires citations.

## Quickstart

```bash
git clone https://github.com/SanaAraj/rag-knowledge-assistant.git
cd rag-knowledge-assistant
pip install -r requirements.txt
```

Add your Groq API key (free at [console.groq.com](https://console.groq.com/keys)) to a `.env` file:

```
GROQ_API_KEY=gsk_your_key_here
```

Add PDFs to `data/`, or try the two included sample documents:

```bash
cp samples/*.pdf data/
```

## Usage

```bash
# Index everything in data/
python main.py --ingest

# Ask a single question (answer is cited to document + page)
python main.py --query "What is supervised learning?"

# Or start an interactive session that loads the model once and stays fast
python main.py --chat
```

Example:

```text
Supervised learning trains a model on labeled examples, where every input is
paired with the correct output, so the model can measure its error directly and
generalize to new inputs. (introduction-to-machine-learning.pdf, Page 2)

--- Sources ---
- introduction-to-machine-learning.pdf, page 2
```

Ask something the documents do not cover and it refuses rather than guessing:

```text
$ python main.py --query "What is the capital of France?"
I don't have enough information in the documents to answer this question.
```

## Tech stack

| Component | Choice |
|---|---|
| Answer generation | Groq (`openai/gpt-oss-20b`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), runs locally |
| Vector store | ChromaDB (persistent, cosine similarity) |
| PDF parsing | PyPDF2 |

## Configuration

Tunable in `config.py`:

- `CHUNK_SIZE` / `CHUNK_OVERLAP` - chunking granularity (words).
- `EMBEDDING_MODEL` - any `sentence-transformers` model.
- the retrieval similarity `threshold` and top-`k` live in `retrieve.py` / `generate.py`.

## Notes and limitations

- **Text-based PDFs only.** Scanned or image-only PDFs need an OCR step first; PyPDF2 extracts embedded text, not pixels.
- **Retrieval quality bounds answer quality.** If the right passage is not retrieved, the answer will not contain it by design; the threshold trades off missed answers against false matches.
- **Cold start.** A one-shot `--query` loads the embedding model each run, so it takes a few seconds before answering; `--chat` loads the model once and then answers quickly.
- **One key required.** Groq is the only external dependency; embeddings and the vector store are fully local.

## Samples

`samples/` contains two generated reference PDFs (machine learning and neural
networks) plus the script that produced them, so the demo above is fully
reproducible.
