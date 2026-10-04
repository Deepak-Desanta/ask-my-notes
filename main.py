from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "documents/Java Notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")


# -----------------------------
# 2. Split PDF into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# 4. Store chunks in Chroma
# -----------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="java_notes"
)

print("Embeddings created and stored!")


# -----------------------------
# 5. Load local LLM
# -----------------------------

llm = OllamaLLM(
    model="qwen2.5:0.5b-instruct"
)


# -----------------------------
# 6. Ask questions
# -----------------------------

while True:

    query = input("\nAsk your PDF (type 'exit' to quit): ")

    if query.lower().strip() == "exit":
        break

    # Retrieve relevant chunks with similarity scores
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=5
    )

    # Check best matching result
    best_score = results[0][1]

    if best_score < 0.4:
        print("\n--- Answer ---")
        print("I couldn't find the answer in your notes.")
        continue

    # Extract retrieved documents
    documents_found = [
        result[0]
        for result in results
    ]

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents_found
    )

    # Create prompt
    prompt = f"""
You answer questions using only the provided PDF context.

Rules:
- Use only information found in the context.
- Do not use outside knowledge.
- Do not add information that is not in the context.
- Do not provide website links.
- Keep the answer clear and concise.
- If the context does not contain the answer, say:
"I couldn't find enough information in your notes."
- For comparison questions, use a simple table if the context supports it.
- Do not contradict the context.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    answer = llm.invoke(prompt)

    print("\n--- Answer ---")
    print(answer)

    # Show source pages
    print("\n--- Sources ---")

    source_pages = []

    for document in documents_found:

        page = document.metadata.get("page")

        if page is not None:
            page = page + 1

            if page not in source_pages:
                source_pages.append(page)

    for page in sorted(source_pages):
        print(f"Page {page}")