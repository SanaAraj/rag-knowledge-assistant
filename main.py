import argparse
from ingest import process_pdfs
from retrieve import add_chunks, clear_collection, get_collection, get_model
from generate import ask


def ingest():
    print("Processing PDFs...")
    clear_collection()
    chunks = process_pdfs()

    if not chunks:
        print("No PDF files found in data/ directory")
        return

    count = add_chunks(chunks)
    print(f"Indexed {count} chunks from {len(set(c['source'] for c in chunks))} document(s)")


def query(question: str):
    coll = get_collection()
    if coll.count() == 0:
        print("No documents indexed. Run with --ingest first.")
        return

    result = ask(question)
    print("\n" + result["answer"])
    if result["sources"]:
        print("\n--- Sources ---")
        print(result["sources"])


def chat():
    """Interactive mode: load the embedding model once, then answer many questions."""
    coll = get_collection()
    if coll.count() == 0:
        print("No documents indexed. Run with --ingest first.")
        return

    print("Loading model...")
    get_model()  # warm up front so every answer is fast
    print("Ready. Ask a question about your documents (empty line to quit).\n")

    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        result = ask(question)
        print("\n" + result["answer"])
        if result["sources"]:
            print("\n--- Sources ---")
            print(result["sources"])
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant")
    parser.add_argument("--ingest", action="store_true", help="Process and index PDFs from data/")
    parser.add_argument("--query", type=str, help="Ask a single question")
    parser.add_argument("--chat", action="store_true", help="Interactive Q&A (loads the model once)")

    args = parser.parse_args()

    if args.ingest:
        ingest()
    elif args.query:
        query(args.query)
    elif args.chat:
        chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
