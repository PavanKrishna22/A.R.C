"""
RAG Engine
Stack: Groq (Llama 3.3-70b) + BAAI/bge-small-en-v1.5 + ChromaDB + BM25
"""

from __future__ import annotations

import os
import io
import re
import tempfile
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from chromadb import Client
from chromadb.config import Settings
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ─────────────────────────────────────────────────
MODEL_ID   = "llama-3.3-70b-versatile"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION  = "ragbot_docs"
TOP_K       = 5
BM25_K      = 3
THRESHOLD   = 0.30   # cosine relevance; below → pure LLM

SYSTEM_RAG = """
You are an expert Retrieval-Augmented Generation (RAG) assistant.

Your primary source of truth is the retrieved document context.

========================
Decision Rules
========================

1. First determine whether the retrieved context is relevant to the user's question.

2. If the context fully answers the question:
   • Answer ONLY using the retrieved information.

3. If the context partially answers the question:
   • Use the document for the supported information.
   • Clearly indicate which parts are not covered.
   • If appropriate, supplement the missing information using your general knowledge.
   • Explicitly distinguish between document information and general knowledge.

4. If the retrieved context is clearly unrelated to the user's question:
   • Ignore the retrieved context.
   • Answer using your own knowledge.
   • Do NOT say "Not found in documents."

5. Never invent facts that appear to come from the documents.

========================
Formatting
========================

• Definitions:
  Start with a one-sentence definition.

• Lists:
  Use markdown bullet points.

• Comparisons:
  Use markdown tables.

• Procedures:
  Use numbered lists.

• Code:
  Use fenced code blocks.

• Bold only important terms.

========================
Style
========================

• Be concise.
• Don't repeat the question.
• No conversational filler.
• Write naturally.
• Prefer readability over verbosity.
"""

SYSTEM_LLM = """
You are an expert AI assistant with deep technical and general knowledge.

Your goal is to provide clear, accurate, well-structured answers.

Rules:

1. Answer immediately.
   - Lead with the answer.
   - Do not repeat the question.
   - Do not use conversational filler such as:
     "Certainly", "Of course", "Great question", etc.

2. Structure your response intelligently.
   - Simple factual questions:
       • 2–5 concise sentences.
   - Definitions:
       • Start with a one-sentence definition.
       • Then provide key details.
   - Comparisons:
       • Prefer markdown tables.
   - Lists:
       • Use bullet points.
   - Instructions:
       • Use numbered steps.
   - Complex topics:
       • Organize using short headings.

3. Markdown formatting.
   - Use **bold** only for important terms.
   - Use tables when useful.
   - Use bullet points instead of long paragraphs.
   - Use code blocks for code.
   - Use inline code for commands, filenames, functions, and variables.

4. Balance detail with brevity.
   - Simple questions:
       • Keep answers under 5 sentences.
   - Complex questions:
       • Expand only when necessary.
   - Avoid unnecessary repetition.

5. If uncertain:
   - State uncertainty honestly.
   - Do not fabricate information.

6. Prioritize readability.
   - Short paragraphs.
   - Logical flow.
   - Clear transitions.
   - High information density.

Your responses should read like those of an experienced engineer or technical writer: concise, structured, informative, and easy to scan.
"""

# ── Routing keywords — no extra LLM call needed ───────────────
_DOC_TRIGGERS = re.compile(
    r"\b(document|file|uploaded|pdf|txt|according to|based on|"
    r"in the|from the|what does it say|summarize|explain the)\b",
    re.I,
)

# ── Text extraction ───────────────────────────────────────────
def _load_text(filename: str, raw: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            pass
    return raw.decode("utf-8", errors="ignore")

# ── Chunking ─────────────────────────────────────────────────
def _recursive_chunks(text: str, size=600, overlap=80) -> list[str]:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(text)

def _semantic_chunks(text: str, max_size=700) -> list[str]:
    """Paragraph-merge semantic chunking — no extra ML model."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= max_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = p if len(p) <= max_size else ""
            if len(p) > max_size:
                sub = _recursive_chunks(p, size=max_size, overlap=60)
                chunks.extend(sub[:-1])
                buf = sub[-1] if sub else ""
    if buf:
        chunks.append(buf)
    return chunks

# ── Engine ────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, chunk_strategy: Literal["recursive", "semantic"] = "recursive"):
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY missing — add it to your .env file.")
        self.groq           = Groq(api_key=api_key)
        self.chunk_strategy = chunk_strategy
        self._all_docs: list[Document] = []   # for BM25
        self._indexed: set[str] = set()

        # BGE embeddings — local, no API key
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: ",
        )

        self._persist_dir = tempfile.mkdtemp(prefix="ragbot_")
        self.vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=self._persist_dir,
        )
        self._bm25: BM25Retriever | None = None

    # ── Ingest ────────────────────────────────────────────────
    def ingest(self, filename: str, raw: bytes) -> int:
        if filename in self._indexed:
            return 0
        text = _load_text(filename, raw)
        if not text.strip():
            return 0

        chunks = (
            _semantic_chunks(text)
            if self.chunk_strategy == "semantic"
            else _recursive_chunks(text)
        )
        docs = [
            Document(page_content=c, metadata={"source": filename, "idx": i})
            for i, c in enumerate(chunks) if c.strip()
        ]
        if not docs:
            return 0

        self.vectorstore.add_documents(docs)
        self._all_docs.extend(docs)
        self._bm25 = BM25Retriever.from_documents(self._all_docs, k=BM25_K)
        self._indexed.add(filename)
        return len(docs)

    def clear_docs(self):
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=self._persist_dir,
        )
        self._all_docs.clear()
        self._bm25 = None
        self._indexed.clear()

    # ── Routing (keyword heuristic — zero latency) ────────────
    def _should_use_rag(self, query: str) -> bool:
        if not self._indexed:
            return False
        # Explicit doc-related language → always RAG
        if _DOC_TRIGGERS.search(query):
            return True
        # Otherwise: try vector search; if top score ≥ threshold → RAG
        hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=1)
        return bool(hits and hits[0][1] >= THRESHOLD)

    # ── Hybrid retrieval (Chroma + BM25, deduplicated) ────────
    def _retrieve(self, query: str) -> tuple[list[Document], list[str]]:
        # Chroma semantic results
        vec_hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)
        vec_docs = [d for d, s in vec_hits if s >= THRESHOLD]

        # BM25 keyword results
        bm25_docs = self._bm25.invoke(query) if self._bm25 else []

        # Merge, deduplicate by content
        seen, merged = set(), []
        for d in vec_docs + bm25_docs:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(d)

        sources = list(dict.fromkeys(d.metadata.get("source", "") for d in merged))
        return merged, sources

    # ── Chat ──────────────────────────────────────────────────
    def chat(self, query: str, history: list[dict], use_rag: bool = True) -> dict:
        mode, sources, context_block = "llm", [], ""

        if use_rag and self._should_use_rag(query):
            docs, sources = self._retrieve(query)
            if docs:
                mode = "rag"
                context_block = "\n\n---\n\n".join(
                    f"[{d.metadata.get('source','doc')}]\n{d.page_content}"
                    for d in docs
                )

        system = SYSTEM_RAG if mode == "rag" else SYSTEM_LLM
        if context_block:
            system += f"\n\n=== CONTEXT ===\n{context_block}\n=== END ==="

        messages = [{"role": "system", "content": system}]
        for m in history[-20:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": query})

        resp = self.groq.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return {
            "answer":  resp.choices[0].message.content,
            "mode":    mode,
            "sources": sources,
        }

    def stream_chat(self, query: str, history: list[dict], use_rag: bool = True):
        """
        Yields: metadata dict first, then token strings.
        First yield: {"mode": ..., "sources": [...]}
        Subsequent yields: str token chunks
        """
        mode, sources, context_block = "llm", [], ""

        if use_rag and self._should_use_rag(query):
            docs, sources = self._retrieve(query)
            if docs:
                mode = "rag"
                context_block = "\n\n---\n\n".join(
                    f"[{d.metadata.get('source','doc')}]\n{d.page_content}"
                    for d in docs
                )

        system = SYSTEM_RAG if mode == "rag" else SYSTEM_LLM
        if context_block:
            system += f"\n\n=== CONTEXT ===\n{context_block}\n=== END ==="

        messages = [{"role": "system", "content": system}]
        for m in history[-20:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": query})

        # Yield metadata first so the UI knows mode/sources before streaming starts
        yield {"mode": mode, "sources": sources}

        stream = self.groq.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
