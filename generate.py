from groq import Groq
from config import GROQ_API_KEY
from retrieve import search, format_sources

client = None


def get_client():
    global client
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)
    return client


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
Rules:
1. Only use information from the provided context to answer
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Cite your sources by mentioning the document name and page number
4. Be concise and direct"""


def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in the documents to answer this question."

    context = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content


def ask(query: str) -> dict:
    """Main function to ask a question and get an answer with sources."""
    chunks = search(query, k=3, threshold=0.3)

    if not chunks:
        return {
            "answer": "I don't have enough information in the documents to answer this question.",
            "sources": ""
        }

    answer = generate_answer(query, chunks)
    sources = format_sources(chunks)

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    result = ask("What are the types of machine learning?")
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    print(result["sources"])
