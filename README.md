# 🤖 RAG Chatbot — Free & Production Grade

**Stack:** Groq (Llama 3.3 70B Versatile) · ChromaDB · HuggingFace Embeddings · Streamlit  
**Cost:** $0 — all components are free to use

---

## Features

| Feature | Detail |
|---|---|
| **LLM Chat** | Llama 3.3 70B via Groq (free tier: 30 req/min) |
| **RAG** | Upload PDFs, TXT, MD, PY, CSV files; answers cite sources |
| **Smart Routing** | LLM decides RAG vs. general-knowledge per question |
| **Chunking** | Recursive (fast) or Semantic (smarter) — switchable |
| **Embeddings** | `all-MiniLM-L6-v2` runs locally, no API key needed |
| **Vector Store** | ChromaDB (in-memory, no server needed) |
| **Chat UI** | Timestamps, response time, source badges on every message |

---

## Quick Start

### 1 — Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2 — Create and set up the project

```bash
# Clone / enter the project folder
cd ragbot

# Initialise uv (creates .venv automatically)
uv sync
```

### 3 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key — starts with `gsk_`

### 4 — Run

```bash
uv run streamlit run app.py
```

The app opens at **http://localhost:8501**  
Paste your Groq API key in the sidebar → start chatting.

---

## uv Commands Reference

```bash
# Install all dependencies
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run any script inside the venv
uv run python rag_engine.py

# Run the app
uv run streamlit run app.py

# Update all packages
uv lock --upgrade
uv sync
```

---

## Project Structure

```
ragbot/
├── app.py            ← Streamlit UI
├── rag_engine.py     ← RAG + Groq logic
├── pyproject.toml    ← uv / pip metadata
├── .env.example      ← API key template
└── README.md
```

---

## How It Works

```
User question
      │
      ▼
[Has indexed docs?] ──No──▶ Pure LLM answer
      │ Yes
      ▼
[Router LLM call]
      │
  RAG ├──▶ Embed query → Chroma similarity search
      │         → Filter by relevance threshold
      │         → Build context → Groq LLM
      │         → Answer + source badges
      │
  LLM └──▶ Groq LLM directly → Answer
```

---

## Chunking Strategies

| Strategy | How | Best For |
|---|---|---|
| **Recursive** | Split on `\n\n`, `\n`, `.`, ` ` | Code, mixed content, fast indexing |
| **Semantic** | Merge paragraphs up to token limit | Prose documents, reports, books |

---

## Tips

- Upload multiple documents — the router figures out which one to use
- Toggle "Auto-RAG" off to force pure LLM mode
- Response time shown below every reply (Groq is fast — typically < 2s)
- Clear documents without restarting: sidebar → "Clear All Documents"
