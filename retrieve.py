import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, EMBEDDING_MODEL

model = None
collection = None


def get_model():
    global model
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def get_collection():
    global collection
    if collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    return collection


def add_chunks(chunks: list[dict]):
    """Add chunks to the vector store."""
    if not chunks:
        return

    coll = get_collection()
    embedder = get_model()

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    coll.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    return len(chunks)


def search(query: str, k: int = 3, threshold: float = 0.7) -> list[dict]:
    """Search for relevant chunks. Returns empty list if no relevant results found."""
    coll = get_collection()
    embedder = get_model()

    if coll.count() == 0:
        return []

    query_embedding = embedder.encode([query]).tolist()

    results = coll.query(
        query_embeddings=query_embedding,
        n_results=min(k, coll.count())
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 1.0
            score = 1 - distance
            if score >= threshold:
                chunks.append({
                    "text": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "page": results["metadatas"][0][i]["page"],
                    "score": score
                })

    return chunks


def format_sources(chunks: list[dict]) -> str:
    """Format chunks with source attribution."""
    if not chunks:
        return ""

    seen = set()
    sources = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"- {c['source']}, page {c['page']}")

    return "\n".join(sources)


def clear_collection():
    """Clear the vector store."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("documents")
    except:
        pass
    global collection
    collection = None


if __name__ == "__main__":
    from ingest import process_pdfs

    clear_collection()
    chunks = process_pdfs()
    print(f"Processing {len(chunks)} chunks...")

    added = add_chunks(chunks)
    print(f"Added {added} chunks to vector store")

    print("\nTesting relevant query...")
    results = search("What is supervised learning?", threshold=0.3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"\n[{r['source']}, p.{r['page']}] (score: {r['score']:.3f})")
        print(r["text"][:150] + "...")

    print("\nSources:")
    print(format_sources(results))

    print("\nTesting irrelevant query...")
    results = search("quantum physics black holes", threshold=0.5)
    print(f"Found {len(results)} results (should be 0 for irrelevant query)")
