"""
RAG Chatbot UI
Stack: Groq · BAAI/bge-small-en-v1.5 · ChromaDB · BM25 · Streamlit
"""
import streamlit as st
import time
import markdown
from datetime import datetime

from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>

/* ---------- Custom Font ---------- */


[data-testid="stAppViewContainer"] {{
    background: #0f1117;
}}

[data-testid="stSidebar"] {{
    background: #161b22;
    border-right: 1px solid #21262d;
}}

[data-testid="stSidebar"] * {{
    color: #c9d1d9 !important;
}}

.msg-wrapper {{
    margin-bottom: 18px;
}}

.msg-user {{
    display: flex;
    justify-content: flex-end;
}}

.msg-assistant {{
    display: flex;
    justify-content: flex-start;
}}

.bubble-user {{
    background: #1f6feb;
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 15px;
    max-width: 72%;
    font-size: .93rem;
    line-height: 1.55;
    word-wrap: break-word;
}}

.bubble-assistant {{
    background: #1c2128;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 18px 18px 18px 4px;
    padding: 11px 15px;
    max-width: 82%;
    font-size: .93rem;
    line-height: 1.55;
    word-wrap: break-word;
}}

.msg-meta {{
    font-size: .70rem;
    color: #484f58;
    margin-top: 3px;
    padding: 0 5px;
}}

.msg-meta-right {{
    text-align: right;
}}

.badge {{
    display: inline-block;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: .68rem;
    margin-top: 5px;
    margin-right: 3px;
}}

.badge-rag {{
    color: #3fb950;
    border: 1px solid #3fb950;
}}

.badge-llm {{
    color: #79c0ff;
    border: 1px solid #79c0ff;
}}

.badge-doc {{
    color: #e3b341;
    border: 1px solid #e3b341;
}}

.sec-label {{
    font-size: .68rem;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 6px;
    padding-left: 2px;
}}

.doc-chip {{
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: .76rem;
    color: #c9d1d9;
    margin: 2px 0;
}}

</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
_DEFAULTS = {
    "messages":       [],
    "engine":         None,
    "docs_loaded":    [],
    "chunk_strategy": "recursive",
    "rag_mode":       True,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Engine factory ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def build_engine(strategy: str) -> RAGEngine:
    """Cached per strategy — avoids reloading the embedding model on every rerun."""
    return RAGEngine(chunk_strategy=strategy)


def get_engine() -> RAGEngine:
    return build_engine(st.session_state.chunk_strategy)

st.set_page_config(
    page_title="A.R.C",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 A.R.C")
    st.caption("Adaptive Reasoning Companion")

    st.markdown("<div class='sec-label'>Chunking Strategy</div>", unsafe_allow_html=True)
    strategy = st.radio(
        "",
        ["recursive", "semantic"],
        index=0 if st.session_state.chunk_strategy == "recursive" else 1,
        format_func=lambda x: {"recursive": "⚡ Recursive (Fast)", "semantic": "🧠 Semantic (Smarter)"}[x],
        label_visibility="collapsed",
    )
    if strategy != st.session_state.chunk_strategy:
        st.session_state.chunk_strategy = strategy
        st.session_state.docs_loaded = []   # old docs belong to old engine
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='sec-label'>Document Upload</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "py", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        engine = get_engine()
        new_files = [f for f in uploaded if f.name not in st.session_state.docs_loaded]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} file(s)…"):
                for f in new_files:
                    engine.ingest(filename=f.name, raw=f.read())
                    st.session_state.docs_loaded.append(f.name)
            st.success(f"✅ {len(new_files)} file(s) indexed")

    if st.session_state.docs_loaded:
        st.markdown("<div class='sec-label' style='margin-top:10px'>Indexed</div>", unsafe_allow_html=True)
        for d in st.session_state.docs_loaded:
            icon = "📄" if d.endswith(".pdf") else "📝"
            st.markdown(f"<div class='doc-chip'>{icon} {d}</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Documents", use_container_width=True):
            get_engine().clear_docs()
            st.session_state.docs_loaded = []
            st.rerun()

    st.markdown("---")
    st.markdown("<div class='sec-label'>Settings</div>", unsafe_allow_html=True)
    st.session_state.rag_mode = st.toggle(
        "Auto-RAG (use docs when relevant)", value=st.session_state.rag_mode
    )

    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.70rem;color:#484f58;text-align:center'>"
        "Llama 3.3 70B · Groq · ChromaDB · BGE · BM25</div>",
        unsafe_allow_html=True,
    )


# ── Chat history ──────────────────────────────────────────────
st.markdown("### 💬 Chat")

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    ts = msg.get("timestamp", "")
    elapsed = msg.get("elapsed")
    mode = msg.get("mode", "llm")
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(f"""
<div class='msg-wrapper'>
  <div class='msg-user'>
    <div class='bubble-user'>{content}</div>
  </div>
  <div class='msg-meta msg-meta-right'>{ts}</div>
</div>
""", unsafe_allow_html=True)

    else:
        mode_badge = (
            "<span class='badge badge-rag'>📚 RAG</span>"
            if mode == "rag"
            else "<span class='badge badge-llm'>🧠 LLM</span>"
        )

        src_badges = "".join(
            f"<span class='badge badge-doc'>📄 {s}</span>"
            for s in sources
        )

        time_str = f" · ⏱ {elapsed:.2f}s" if elapsed else ""

        formatted_content = markdown.markdown(
            content,
            extensions=["fenced_code", "tables"]
        )

        st.markdown(f"""
<div class='msg-wrapper'>
  <div class='msg-assistant'>
    <div class='bubble-assistant'>
        {formatted_content}
        <div style='margin-top:7px'>
            {mode_badge}{src_badges}
        </div>
    </div>
  </div>
  <div class='msg-meta'>
      {ts}{time_str}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything…")

if prompt:
    try:
        engine = get_engine()
    except EnvironmentError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", "content": prompt, "timestamp": now,
    })
    # Show the user bubble immediately
    st.markdown(f"""
<div class='msg-wrapper'>
  <div class='msg-user'><div class='bubble-user'>{prompt}</div></div>
  <div class='msg-meta msg-meta-right'>{now}</div>
</div>""", unsafe_allow_html=True)

    # Stream the assistant response
    use_rag = st.session_state.rag_mode and bool(st.session_state.docs_loaded)
    stream  = engine.stream_chat(
        query=prompt,
        history=st.session_state.messages[:-1],
        use_rag=use_rag,
    )

    # First yield is always the metadata dict
    t0       = time.perf_counter()
    metadata = next(stream)
    mode     = metadata["mode"]
    sources  = metadata["sources"]

    mode_badge = (
        "<span class='badge badge-rag'>📚 RAG</span>"
        if mode == "rag"
        else "<span class='badge badge-llm'>🧠 LLM</span>"
    )
    src_badges = "".join(
        f"<span class='badge badge-doc'>📄 {s}</span>" for s in sources
    )

    # Stream tokens — batch updates every 10 tokens for speed
    placeholder = st.empty()
    accumulated = ""
    buffer = ""
    token_count = 0
    FLUSH_EVERY = 8   # update UI every N tokens

    for token in stream:
        buffer += token
        token_count += 1

        if token_count % FLUSH_EVERY == 0:

            accumulated += buffer
            buffer = ""

            formatted_stream = markdown.markdown(
                accumulated + "▌",
                extensions=["fenced_code", "tables"]
            )

            placeholder.markdown(f"""
    <div class='msg-wrapper'>
    <div class='msg-assistant'>
        <div class='bubble-assistant'>
            {formatted_stream}
            <div style='margin-top:7px'>
                {mode_badge}{src_badges}
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # flush remaining buffer
    accumulated += buffer

    elapsed = time.perf_counter() - t0
    reply_ts = datetime.now().strftime("%H:%M:%S")

    formatted_reply = markdown.markdown(
        accumulated,
        extensions=["fenced_code", "tables"]
    )

    placeholder.markdown(f"""
    <div class='msg-wrapper'>
    <div class='msg-assistant'>
        <div class='bubble-assistant'>
            {formatted_reply}
            <div style='margin-top:7px'>
                {mode_badge}{src_badges}
            </div>
        </div>
    </div>
    <div class='msg-meta'>
        {reply_ts} · ⏱ {elapsed:.2f}s
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   accumulated,
        "timestamp": reply_ts,
        "elapsed":   elapsed,
        "mode":      mode,
        "sources":   sources,
    })
