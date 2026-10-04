# Ask My Notes

A local Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a PDF and receive answers based on its contents.

The project uses local AI models through Ollama, so no OpenAI API key or paid API is required.

## Features

- Load and process a PDF document
- Split the document into smaller text chunks
- Generate embeddings locally
- Store embeddings using Chroma
- Retrieve relevant document chunks using similarity search
- Use a local LLM to generate answers
- Reject questions that are unrelated to the document
- Display the source pages used for an answer
- Interactive terminal-based question answering

## RAG Pipeline

```text
PDF
 ↓
Load document
 ↓
Split into chunks
 ↓
Generate embeddings
 ↓
Store in Chroma
 ↓
User question
 ↓
Similarity search
 ↓
Retrieve relevant chunks
 ↓
Send context to LLM
 ↓
Generate answer
 ↓
Display source pages