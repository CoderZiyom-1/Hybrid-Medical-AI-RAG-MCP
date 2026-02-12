DOC SAABH — Hybrid Medical AI Assistant (RAG + MCP)
Overview

DOC SAABH is an advanced AI-powered medical assistant that combines:

Retrieval-Augmented Generation (RAG)

Local LLMs (Mistral via Ollama)

FAISS Vector Database

MCP Tool Integration

to provide accurate, document-grounded medical responses.

Unlike traditional chatbots, this system:

1. Answers ONLY from medical documents
2. Avoids hallucinations
3. Supports external knowledge via MCP tools
4. Runs fully locally (privacy-friendly)

**Tech Stack**

LangChain

FAISS Vector DB

Ollama (Mistral LLM)

Streamlit

MCP (Model Context Protocol)

**Architecture**

1. PDF Loader → Extracts medical knowledge

2. Text Chunking → Splits into embeddings

3. FAISS → Stores semantic vectors

4. RAG → Retrieves relevant medical context

5. MCP → Fetches external verified info

6. LLM → Generates grounded responses


**Features**

Local AI inference

Document-grounded responses

Hybrid reasoning (RAG + MCP)

Medical hallucination control

Streamlit chat interface
