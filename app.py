"""
DocuMind AI — Retrieval-Augmented Generation Web App
MCA Prompt Engineering Assignment | Chanakya University

Run:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
# NEW — works both locally AND on Streamlit Cloud
from dotenv import load_dotenv
load_dotenv()

# Streamlit Cloud secrets support
try:
    import streamlit as st
    if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass
# ── Page config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy imports (speeds up initial render) ──────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_modules():
    from rag.pdf_loader import load_document
    from rag.chunker import chunk_text
    from rag.embeddings import embed_texts, get_embed_model
    from rag.vector_store import add_chunks, get_chunk_count, clear_collection
    from rag.rag_pipeline import run_rag
    # Warm up embedding model
    get_embed_model()
    return load_document, chunk_text, embed_texts, add_chunks, get_chunk_count, clear_collection, run_rag


# ── CSS ──────────────────────────────────────────────────────────
def load_css(dark: bool = False):
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            base_css = f.read()
    else:
        base_css = ""

    dark_css = ""
    if dark:
        dark_css = """
        /* ── Dark Mode — Complete Override ── */

        /* CSS token reset */
        :root {
          --bg: #0d1117 !important;
          --surface: #161b22 !important;
          --border: #30363d !important;
          --text: #e6edf3 !important;
          --text-muted: #8b949e !important;
          --primary: #58a6ff !important;
          --primary-light: #79b8ff !important;
          --primary-faint: #0d2542 !important;
          --accent: #bc8cff !important;
          --success: #3fb950 !important;
          --shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
          --shadow-md: 0 4px 12px rgba(0,0,0,0.4) !important;
        }

        /* App background */
        .stApp,
        .stApp > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="block-container"] {
          background-color: #0d1117 !important;
          color: #e6edf3 !important;
        }

        /* Main content area */
        .main .block-container {
          background-color: #0d1117 !important;
        }

        /* ALL text to light color */
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp li, .stApp a,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {
          color: #e6edf3 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] .block-container {
          background-color: #161b22 !important;
          border-right: 1px solid #30363d !important;
        }
        section[data-testid="stSidebar"] * {
          color: #e6edf3 !important;
        }

        /* Header */
        .dm-header {
          background: #161b22 !important;
          border-bottom-color: #30363d !important;
        }
        .dm-title { color: #e6edf3 !important; }
        .dm-subtitle { color: #8b949e !important; }

        /* Buttons */
        .stButton > button {
          background-color: #21262d !important;
          color: #e6edf3 !important;
          border: 1px solid #30363d !important;
        }
        .stButton > button:hover {
          background-color: #30363d !important;
          border-color: #58a6ff !important;
          color: #58a6ff !important;
        }

        /* Chat input box */
        .stChatInput,
        .stChatInput > div,
        .stChatInput textarea,
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInputContainer"],
        [data-testid="stChatInputContainer"] > div {
          background-color: #161b22 !important;
          border-color: #30363d !important;
          color: #e6edf3 !important;
        }
        .stChatInput textarea::placeholder {
          color: #8b949e !important;
        }

        /* Chat messages */
        [data-testid="stChatMessage"],
        [data-testid="stChatMessageContent"],
        .stChatMessage {
          background-color: #161b22 !important;
          border-color: #30363d !important;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] span {
          color: #e6edf3 !important;
        }

        /* Expanders */
        [data-testid="stExpander"],
        .streamlit-expanderHeader,
        .streamlit-expanderContent {
          background-color: #161b22 !important;
          border-color: #30363d !important;
          color: #e6edf3 !important;
        }
        .streamlit-expanderHeader:hover {
          background-color: #21262d !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {
          color: #e6edf3 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploaderDropzone"] {
          background-color: #161b22 !important;
          border-color: #58a6ff !important;
          color: #e6edf3 !important;
        }
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] small {
          color: #8b949e !important;
        }

        /* Toggle widget */
        [data-testid="stToggle"] span {
          color: #e6edf3 !important;
        }

        /* Alerts / info boxes */
        [data-testid="stAlert"],
        .stAlert {
          background-color: #161b22 !important;
          border-color: #30363d !important;
          color: #e6edf3 !important;
        }

        /* Spinner */
        [data-testid="stSpinner"] p { color: #8b949e !important; }

        /* Divider */
        hr { border-color: #30363d !important; }

        /* Custom cards */
        .sidebar-section {
          background: #0d1117 !important;
          border-color: #30363d !important;
        }
        .sidebar-label { color: #8b949e !important; }
        .stat-row { border-bottom-color: #30363d !important; color: #e6edf3 !important; }
        .stat-val { color: #58a6ff !important; }

        .source-card {
          background: #161b22 !important;
          border-color: #30363d !important;
          border-left-color: #58a6ff !important;
        }
        .source-header { color: #58a6ff !important; }
        .source-text { color: #8b949e !important; }

        /* Welcome screen text */
        .stApp [style*="color: #94a3b8"],
        .stApp [style*="color: #334155"],
        .stApp [style*="color: #475569"] {
          color: #8b949e !important;
        }
        """

    st.markdown(f"<style>{base_css}\n{dark_css}</style>", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],         # [{role, content}]
        "display_history": [],      # [{role, content, sources}]
        "indexed_files": set(),     # filenames already indexed
        "dark_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ── Apply CSS with current theme ─────────────────────────────────
load_css(dark=st.session_state.dark_mode)

# ── Header ───────────────────────────────────────────────────────
api_ok = bool(os.getenv("OPENROUTER_API_KEY", "").strip())

st.markdown(
    f"""
    <div class="dm-header">
      <div class="dm-logo">🧠</div>
      <div>
        <div class="dm-title">DocuMind AI</div>
        <div class="dm-subtitle">AI-Powered Academic Knowledge Assistant</div>
      </div>
      <div class="dm-badge">{'AI Ready' if api_ok else '⚠ No API Key'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ DocuMind AI")
    st.markdown("---")

    # Dark mode toggle
    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.markdown("---")

    # Upload section
    st.markdown('<div class="sidebar-label">📂 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & drop PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload your study material, notes, or any document.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        # Load RAG modules (cached after first call)
        with st.spinner("Loading AI components…"):
            load_doc, chunk_text, embed_texts, add_chunks, get_count, clear_col, run_rag = load_rag_modules()

        for uf in uploaded_files:
            if uf.name in st.session_state.indexed_files:
                st.success(f"✓ {uf.name} already indexed")
                continue

            with st.spinner(f"Indexing {uf.name}…"):
                try:
                    # Save to temp file
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name

                    # Extract → chunk → embed → store
                    text = load_doc(tmp_path)
                    chunks = chunk_text(text)
                    embeddings = embed_texts(chunks)
                    new_added = add_chunks(chunks, embeddings, uf.name)

                    os.unlink(tmp_path)
                    st.session_state.indexed_files.add(uf.name)
                    st.success(f"✅ {uf.name} — {new_added} chunks indexed")

                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")

    st.markdown("---")

    # Stats
    try:
        _, _, _, _, get_count, _, _ = load_rag_modules()
        chunk_count = get_count()
    except Exception:
        chunk_count = 0

    st.markdown('<div class="sidebar-label">📊 System Status</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-section">
          <div class="stat-row"><span>Indexed Chunks</span><span class="stat-val">{chunk_count}</span></div>
          <div class="stat-row"><span>Files Loaded</span><span class="stat-val">{len(st.session_state.indexed_files)}</span></div>
          <div class="stat-row"><span>Chat Turns</span><span class="stat-val">{len(st.session_state.display_history)}</span></div>
          <div class="stat-row"><span>API Status</span><span class="stat-val">{'✅' if api_ok else '❌'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

    # Clear knowledge base
    if st.button("🔄 Reset Knowledge Base", use_container_width=True):
        try:
            _, _, _, _, _, clear_col, _ = load_rag_modules()
            clear_col()
        except Exception:
            pass
        st.session_state.indexed_files = set()
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.success("Knowledge base cleared.")
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-label">🛠 Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-section" style="font-size:12px; color: var(--text-muted, #64748b);">
          🔵 Streamlit &nbsp;|&nbsp; ChromaDB<br>
          🟢 Sentence Transformers<br>
          🟣 OpenRouter (LLaMA 3 8B)<br>
          🟠 PyPDF &nbsp;|&nbsp; python-dotenv
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Chat Area ───────────────────────────────────────────────
col_main, col_side = st.columns([3, 1], gap="large")

with col_main:
    # Welcome state
    if not st.session_state.display_history:
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
              <div style="font-size: 52px; margin-bottom: 16px;">🧠</div>
              <div style="font-size: 22px; font-weight: 600; color: #334155; margin-bottom: 8px;">
                Welcome to DocuMind AI
              </div>
              <div style="font-size: 15px; max-width: 420px; margin: 0 auto; line-height: 1.6;">
                Upload your documents from the sidebar, then ask questions.<br>
                All answers are grounded strictly in your documents.
              </div>
              <div style="margin-top: 28px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
                <span style="background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; border-radius:20px; padding:6px 14px; font-size:13px;">📄 Upload PDF / TXT</span>
                <span style="background:#f5f3ff; color:#7c3aed; border:1px solid #ddd6fe; border-radius:20px; padding:6px 14px; font-size:13px;">🔍 Semantic Search</span>
                <span style="background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; border-radius:20px; padding:6px 14px; font-size:13px;">✅ Grounded Answers</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render chat history
    for turn in st.session_state.display_history:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(turn["content"])

                # Show retrieved sources
                if turn.get("sources"):
                    with st.expander(f"📚 Retrieved Sources ({len(turn['sources'])} chunks)", expanded=False):
                        for i, src in enumerate(turn["sources"], 1):
                            score_pct = int(src["score"] * 100)
                            color = "#059669" if score_pct >= 80 else "#d97706" if score_pct >= 60 else "#64748b"
                            st.markdown(
                                f"""
                                <div class="source-card">
                                  <div class="source-header">
                                    📄 {src['source']}
                                    <span class="score-badge" style="background: {color}18; color: {color};">
                                      {score_pct}% match
                                    </span>
                                  </div>
                                  <div class="source-text">{src['text'][:350]}{'…' if len(src['text']) > 350 else ''}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat input
    user_input = st.chat_input("Ask a question about your documents…")

    if user_input:
        # Validate pre-conditions
        if not api_ok:
            st.error("⚠️ No API key found. Add OPENROUTER_API_KEY to your .env file and restart.")
            st.stop()

        try:
            _, _, _, _, get_count, _, run_rag = load_rag_modules()
        except Exception as e:
            st.error(f"Failed to load RAG components: {e}")
            st.stop()

        if get_count() == 0:
            st.warning("📂 No documents indexed yet. Please upload files from the sidebar first.")
            st.stop()

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                answer, sources = run_rag(
                    user_input,
                    st.session_state.chat_history,
                    top_k=5,
                )

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Retrieved Sources ({len(sources)} chunks)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        score_pct = int(src["score"] * 100)
                        color = "#059669" if score_pct >= 80 else "#d97706" if score_pct >= 60 else "#64748b"
                        st.markdown(
                            f"""
                            <div class="source-card">
                              <div class="source-header">
                                📄 {src['source']}
                                <span class="score-badge" style="background: {color}18; color: {color};">
                                  {score_pct}% match
                                </span>
                              </div>
                              <div class="source-text">{src['text'][:350]}{'…' if len(src['text']) > 350 else ''}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Update state
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.display_history.append({"role": "user", "content": user_input})
        st.session_state.display_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# ── Right column: Quick info ─────────────────────────────────────
with col_side:
    if not api_ok:
        st.markdown(
            """
            <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:12px; padding:16px; font-size:13px; color:#991b1b;">
              <strong>⚠️ API Key Missing</strong><br><br>
              Create a <code>.env</code> file in the project root:<br><br>
              <code>OPENROUTER_API_KEY=your_key_here</code><br><br>
              Get a free key at <a href="https://openrouter.ai" target="_blank">openrouter.ai</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:16px; font-size:13px; color:#475569; margin-top:12px;">
          <strong style="color:#0f172a;">💡 How It Works</strong>
          <ol style="margin:10px 0 0 0; padding-left:18px; line-height:1.8;">
            <li>Upload PDF/TXT documents</li>
            <li>Text is chunked & embedded</li>
            <li>Stored in ChromaDB vector store</li>
            <li>Your question is semantically matched</li>
            <li>Top chunks sent to LLaMA 3</li>
            <li>Grounded answer returned</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    