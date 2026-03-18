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


def search(query: str, k: int = 3) -> list[dict]:
    """Search for relevant chunks."""
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
            chunks.append({
                "text": doc,
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i] if results["distances"] else None
            })

    return chunks


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

    print("\nTesting search...")
    results = search("What is supervised learning?")
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"\n[{r['source']}, p.{r['page']}] (score: {1 - r['distance']:.3f})")
        print(r["text"][:150] + "...")
