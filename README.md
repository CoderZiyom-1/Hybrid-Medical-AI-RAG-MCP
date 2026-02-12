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

**Setup Instructions**

1. Install Ollama
https://ollama.com


Pull model:

ollama pull mistral

2. Install dependencies
pip install -r requirements.txt

3. Ingest PDFs

Update path inside:

data_ingestion.py


Then run:

python data_ingestion.py

4. Run App
streamlit run app.py

**Use Cases**

Clinical decision support

Medical Q&A from PDFs

Offline hospital AI assistant

Research document analysis

Future Improvements

Multi-document support

Voice input

Real-time EHR integration

Author

Moyiz khan




Then add to README under Demo section:

![Demo](screenshots/demo.png)

