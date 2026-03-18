import argparse
from ingest import process_pdfs
from retrieve import add_chunks, clear_collection, get_collection
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


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant")
    parser.add_argument("--ingest", action="store_true", help="Process and index PDFs from data/")
    parser.add_argument("--query", type=str, help="Ask a question")

    args = parser.parse_args()

    if args.ingest:
        ingest()
    elif args.query:
        query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
