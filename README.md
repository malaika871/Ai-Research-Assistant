---
title: NexusAI
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# AI Research Assistant

## Purpose

This project is an AI-powered research assistant designed to help users interact with academic and technical documents more effectively. Instead of reading every paper manually, users can upload documents such as PDFs, TXT files, and DOCX files, then ask questions in natural language and receive grounded answers based on the uploaded content.

## What the solution does

The system combines document ingestion, chunking, embedding, retrieval, and answer generation into a single workflow:

1. Upload research documents.
2. Extract and split the text into meaningful chunks.
3. Convert those chunks into vector embeddings.
4. Retrieve the most relevant chunks for a question.
5. Generate an answer grounded in those retrieved passages.

This makes the application especially useful for reading, reviewing, and synthesizing research material quickly.

## Why this is a research assistant

This is a research assistant because it is specifically built to support the research workflow rather than act as a general chatbot.

It helps researchers by:

- turning a collection of papers into a searchable knowledge base
- allowing fast question answering over long documents
- grounding responses in source passages instead of guessing
- making it easier to locate relevant sections of a document
- reducing the time spent manually scanning large amounts of text

In other words, it does not just answer questions generally; it helps users work with research content in a focused, evidence-based way.

## Approach

The project uses a Retrieval-Augmented Generation (RAG) approach.

### How it works

- Documents are parsed and split into smaller chunks.
- Each chunk is embedded into a vector space.
- When a user asks a question, the system embeds the question and retrieves the most relevant chunks.
- Those retrieved chunks are passed to a language model as context.
- The model generates an answer using only that context, which improves relevance and reduces hallucinations.

This approach is well suited to research because it anchors answers to the documents the user actually uploaded.

## Tech stack

- Python
- FastAPI for the backend API
- Next.js for the frontend interface
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Ollama / Hugging Face-based LLM integration for generation
- PyMuPDF, python-docx, and text loaders for document parsing
- NumPy and scikit-learn for supporting processing tasks

## Benefits

- Faster literature review and document exploration
- Better question answering over long documents
- Source-grounded responses that are easier to trust
- Flexible support for multiple document formats
- Simple upload-and-query workflow for non-technical users
- Useful for students, researchers, and analysts working with technical material

## Example use cases

- Reviewing academic papers
- Summarizing technical reports
- Extracting key insights from uploaded research materials
- Comparing concepts across multiple documents
- Asking follow-up questions about a specific paper or dataset

## Getting started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. Start the frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Configuration

Copy `src/.env.example` to `src/.env` and fill in your `HF_TOKEN`. All other variables have sensible defaults for local development — see the comments in `.env.example` for what each one controls (CORS origins, API key auth, upload limits, rate limits, model/generation settings).

## Running in production

1. Set environment variables (at minimum):
   ```bash
   ENV=production
   REQUIRE_API_KEY=true
   API_KEYS=<a long random secret>
   CORS_ORIGINS=https://your-frontend-domain.com
   ```

2. Build and run with Docker:
   ```bash
   docker build -t ai-research-assistant .
   ```

3. Clients must send the API key on every request:
   ```bash
   curl -H "X-API-Key: <your key>" https://your-api-domain.com/documents
   ```
   
**Note:** `app.py` (the Gradio UI) is a separate local demo interface, not part of the production API path — the FastAPI app in `backend/main.py` is the deployable service the Next.js frontend talks to.

## Summary

This project is a practical research assistant that helps users move from raw documents to useful insights. By combining retrieval and generation, it enables a more precise and evidence-based way to explore research material.