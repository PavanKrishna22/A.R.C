# A.R.C 🤖

### **Adaptive Reasoning Companion**

*A fast, production-ready Retrieval-Augmented Generation (RAG) chatbot powered by Groq, ChromaDB, BGE Embeddings, and Streamlit.*

---

## Overview

A.R.C (Adaptive Reasoning Companion) is an intelligent chatbot capable of answering both **general knowledge** questions and **document-specific** questions.

Unlike traditional RAG applications that always search uploaded documents, A.R.C first determines whether a question should be answered from the uploaded knowledge base or from the LLM's own knowledge. This results in more natural conversations while still providing grounded answers when documents are relevant.

The project is built entirely with **free and open-source technologies**, making it easy to run locally or deploy on Streamlit Community Cloud.

---

# Demo

> *(Add screenshots or GIFs here)*

```
📄 Upload Documents
        ↓
🔍 Hybrid Retrieval (Chroma + BM25)
        ↓
🧠 Llama 3.3 70B (Groq)
        ↓
💬 Beautiful Streamlit Chat Interface
```

---

# ✨ Features

| Feature                                     | Detail                                                                                                                |
| :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------- |
| **💬 LLM Chat**                             | Powered by **Llama 3.3 70B Versatile** through **Groq** for ultra-fast inference.                                     |
| **📚 Retrieval-Augmented Generation (RAG)** | Upload **PDF, TXT, Markdown, Python, and CSV** files and receive grounded answers from your documents.                |
| **🧠 Intelligent Routing**                  | Automatically decides whether to answer from uploaded documents or general LLM knowledge.                             |
| **⚡ Hybrid Retrieval**                      | Combines **dense semantic search (ChromaDB + BGE embeddings)** with **BM25 keyword retrieval** for improved accuracy. |
| **✂️ Multiple Chunking Strategies**         | Choose between **Recursive Character Chunking** (fast) or **Semantic Paragraph Chunking** (higher quality).           |
| **🔍 Local Embeddings**                     | Uses **BAAI/bge-small-en-v1.5** locally—no embedding API or internet connection required.                             |
| **🗂️ Vector Database**                     | Uses **ChromaDB** with local persistence. No external vector database required.                                       |
| **📄 Source Attribution**                   | Displays the source document(s) used to generate each RAG response.                                                   |
| **⚡ Streaming Responses**                   | Tokens stream in real time for a smooth ChatGPT-like experience.                                                      |
| **🎨 Modern UI**                            | Chat bubbles, timestamps, response times, and source badges.                                                          |
| **🛡️ Secure Deployment**                   | API keys are stored using **Streamlit Secrets** and are never exposed in the repository.                              |

---

# 🏗 System Architecture

```text
                          ┌─────────────────────┐
                          │      User Query     │
                          └──────────┬──────────┘
                                     │
                           Smart Query Router
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
          General Knowledge                       Document Query
                 │                                       │
                 ▼                                       ▼
         Llama 3.3 (Groq)                     Hybrid Retrieval
                                       ┌──────────┴──────────┐
                                       │                     │
                                ChromaDB Search        BM25 Search
                                       │                     │
                                       └──────────┬──────────┘
                                                  │
                                           Context Builder
                                                  │
                                                  ▼
                                         Llama 3.3 (Groq)
                                                  │
                                                  ▼
                                          Streamlit Interface
```

---

# Tech Stack

| Component              | Technology              |
| ---------------------- | ----------------------- |
| **Frontend**           | Streamlit               |
| **LLM**                | Llama 3.3 70B Versatile |
| **Inference Provider** | Groq                    |
| **Embeddings**         | BAAI/bge-small-en-v1.5  |
| **Vector Database**    | ChromaDB                |
| **Keyword Search**     | BM25                    |
| **Chunking**           | Recursive & Semantic    |
| **PDF Parsing**        | PyPDF                   |
| **Language Framework** | LangChain               |
| **Package Manager**    | uv                      |
| **Language**           | Python 3.11+            |

---

# Project Structure

```
A.R.C/
│
├── app.py                  # Streamlit UI
├── rag_engine.py           # Core RAG pipeline
├── app_v2.py               # Alternate UI/version
│
├── sample.txt              # Sample document
│
├── pyproject.toml          # Dependencies
├── uv.lock                 # Locked dependency versions
│
├── .gitignore
├── README.md
└── .env                    # Local secrets (ignored by Git)
```

---

# How It Works

### 1. Upload Documents

Supported formats:

* PDF
* TXT
* Markdown
* Python
* CSV

---

### 2. Document Processing

Documents are

* parsed
* cleaned
* chunked

using one of two strategies:

* Recursive Character Chunking
* Semantic Paragraph Chunking

---

### 3. Embedding Generation

Each chunk is embedded using

**BAAI/bge-small-en-v1.5**

Embeddings run locally on your machine.

---

### 4. Hybrid Retrieval

When a document question is detected:

* Semantic Search (ChromaDB)
* Keyword Search (BM25)

are combined into a single ranked context.

---

### 5. Response Generation

The retrieved context is sent to

**Llama 3.3 70B**

through the **Groq API**, which generates the final response.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/PavanKrishna22/A.R.C.git

cd A.R.C
```

---

## Install uv

If you don't already have **uv** installed:

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Create the Virtual Environment

```bash
uv venv
```

Activate it.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
uv sync
```

---

# Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Running the Application

```bash
streamlit run app.py
```

The application will launch at

```
http://localhost:8501
```

---

# Deploying to Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new app on **Streamlit Community Cloud**.
3. Select your repository.
4. Set the main file to:

```
app.py
```

5. Add your secret:

```toml
GROQ_API_KEY="your_api_key"
```

6. Deploy.

Your API key remains encrypted and is **never exposed** in your repository.

---

# Supported File Types

| Extension | Supported |
| --------- | --------- |
| PDF       | ✅         |
| TXT       | ✅         |
| Markdown  | ✅         |
| Python    | ✅         |
| CSV       | ✅         |

---

# Models Used

## Large Language Model

```
Llama 3.3 70B Versatile
```

Provider:

```
Groq
```

---

## Embedding Model

```
BAAI/bge-small-en-v1.5
```

Runs locally and requires no API key.

---

## Retrieval

* ChromaDB Vector Search
* BM25 Keyword Search

---

# Future Enhancements

Planned improvements include:

### 🖼️ Vision & Multimodal RAG

* Image embeddings from PDFs
* OCR support for scanned documents
* Vision-language models for image understanding
* Diagram and figure-aware retrieval

### 📊 Table-Aware Retrieval

* Preserve table structure during ingestion
* Better retrieval from spreadsheets and PDF tables
* Native support for complex tabular documents

### 🌐 Web Search Agent

* Live web search fallback
* Source citations for online content
* Hybrid RAG + Web Search workflow

### 🌍 Multilingual Support

* Multilingual embeddings
* Cross-language retrieval
* Responses in multiple languages

### 🧠 Advanced Retrieval

* Contextual Compression Retriever
* Parent Document Retriever
* Multi-Query Retrieval
* HyDE Retrieval
* Cross-Encoder Reranking

### 📑 Better Document Processing

* Metadata-aware retrieval
* Page-level citations
* Section-aware chunking
* Larger document support

### 🤖 Agentic Capabilities

* Tool calling
* Function calling
* Multi-step reasoning
* Long-term memory
* Code execution

### ☁️ Production Features

* Persistent document storage
* Multi-user support
* Authentication
* Conversation history
* Document collections

### ⚡ Performance Improvements

* GPU embedding support
* Faster indexing
* Async retrieval pipeline
* Incremental document updates

---

# Contributing

Contributions are welcome!

If you have ideas for improving retrieval quality, UI/UX, performance, or adding new capabilities, feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the **MIT License**.

---

# Author

**Pavan Krishna**

If you found this project useful, consider giving it a ⭐ on GitHub!
