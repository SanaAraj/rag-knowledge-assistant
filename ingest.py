import os
from PyPDF2 import PdfReader
from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF with page numbers."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text,
                "page": i + 1,
                "source": os.path.basename(pdf_path)
            })
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def process_pdfs(data_path: str = DATA_PATH) -> list[dict]:
    """Process all PDFs in the data directory."""
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        return []

    all_chunks = []
    for filename in os.listdir(data_path):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(data_path, filename)
        pages = extract_text_from_pdf(pdf_path)

        for page_data in pages:
            chunks = chunk_text(page_data["text"])
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": page_data["source"],
                    "page": page_data["page"]
                })

    return all_chunks


if __name__ == "__main__":
    chunks = process_pdfs()
    print(f"Processed {len(chunks)} chunks")
    if chunks:
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} (Source: {chunk['source']}, Page: {chunk['page']}) ---")
            print(chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"])
