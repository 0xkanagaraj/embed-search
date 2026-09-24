"""
app.py — SemanticSearch  (redesigned UI + bug fixes)

Bug fixes vs original:
  1. Re-upload duplicate vectors: remove_file() is now called before index_file()
     so stale vectors are cleared before re-indexing the same filename.
  2. llm.py resp.text None: guarded in llm.py (see that file).
  3. store.py unused import pickle: removed in that file.
  4. Empty-field validation added to login / signup forms.
  5. add_file() flow simplified — single upsert after indexing.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from auth import signup, login
from db import add_file, list_files, delete_file
from pipeline import index_file, remove_file, retrieve, build_context
from llm import generate

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SemanticSearch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FILES_ROOT = Path("data/users")

# ─────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────
STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
      rel="stylesheet">
<style>
/* ── Typography base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── App background ── */
.stApp { background: #f0f4f8; }
.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid #1a2744;
    min-width: 290px;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1rem 1.5rem;
}

/* ── Sidebar typography ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #8899b4 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #dde5f0 !important; }
[data-testid="stSidebar"] hr {
    border-color: #1a2744 !important;
    margin: 0.6rem 0 !important;
}

/* ── Sidebar checkboxes ── */
[data-testid="stSidebar"] .stCheckbox label {
    color: #b8c8de !important;
    font-size: 0.82rem !important;
    font-weight: 500;
}

/* ── Sidebar radio ── */
[data-testid="stSidebar"] .stRadio label {
    color: #b8c8de !important;
    font-size: 0.84rem !important;
}

/* ── Sidebar metrics ── */
[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: #121c32;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 9px 14px;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #5a7098 !important; font-size: 0.72rem !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #dde5f0 !important; font-size: 1.25rem !important; }

/* ── Sidebar buttons (default) ── */
[data-testid="stSidebar"] .stButton > button {
    background: #121c32;
    border: 1px solid #1e3050;
    color: #8899b4 !important;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    width: 100%;
    padding: 7px 12px;
    transition: background 0.15s, border-color 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a2744;
    border-color: #2a4070;
    color: #dde5f0 !important;
}

/* ── File uploader (sidebar) ── */
[data-testid="stFileUploader"] {
    background: #121c32;
    border: 2px dashed #1e3050;
    border-radius: 10px;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] small { color: #5a7098 !important; font-size: 0.79rem !important; }

/* ── Primary button ── */
button[kind="primary"] {
    background: #2563eb !important;
    border: none !important;
    border-radius: 9px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    letter-spacing: -0.01em;
    transition: background 0.15s, box-shadow 0.15s !important;
}
button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.09);
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e4ebf5;
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem;
    color: #4a6080;
    padding: 8px 14px;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Tabs (auth) ── */
.stTabs [data-baseweb="tab-highlight"] { background: #2563eb !important; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.875rem;
    font-weight: 600;
    color: #64748b;
}
.stTabs [aria-selected="true"] { color: #1e293b !important; }

/* ── Code blocks ── */
.stCode { font-size: 0.78rem !important; }

/* ─────────── Custom HTML components ─────────── */

/* Logo bar */
.sbar-logo {
    display: flex;
    align-items: center;
    gap: 9px;
    padding-bottom: 18px;
    margin-bottom: 4px;
    border-bottom: 1px solid #1a2744;
}
.sbar-logo-mark {
    width: 32px; height: 32px;
    background: #2563eb;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}
.sbar-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #dde5f0;
    letter-spacing: -0.02em;
}
.sbar-logo-tag {
    margin-left: auto;
    font-size: 0.58rem;
    font-weight: 700;
    background: #162550;
    color: #60a5fa;
    padding: 2px 6px;
    border-radius: 5px;
    letter-spacing: 0.06em;
}

/* User badge */
.sbar-user {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #121c32;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 9px 12px;
    margin: 14px 0 6px;
}
.sbar-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
}
.sbar-uname { font-size: 0.86rem; font-weight: 600; color: #dde5f0; line-height: 1.3; }
.sbar-ustatus { font-size: 0.7rem; color: #2563eb; font-weight: 500; }

/* Section labels */
.sbar-section {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #3a5278;
    margin: 18px 0 8px 2px;
}

/* No-files hint */
.sbar-empty {
    font-size: 0.8rem;
    color: #3a5278;
    padding: 6px 2px;
}

/* Main header card */
.main-hdr {
    background: white;
    border-radius: 16px;
    padding: 16px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.main-hdr-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0b1120;
    letter-spacing: -0.02em;
    margin: 0;
}
.main-hdr-sub {
    font-size: 0.79rem;
    color: #64748b;
    margin: 3px 0 0;
    font-weight: 400;
}
.mode-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 0.77rem;
    font-weight: 700;
    white-space: nowrap;
}
.files-count {
    font-size: 0.71rem;
    color: #94a3b8;
    margin-top: 5px;
    text-align: right;
    font-weight: 500;
}

/* Empty-chat state */
.empty-chat {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 56px 0 40px;
    gap: 8px;
}
.empty-chat-icon { font-size: 3rem; }
.empty-chat-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
}
.empty-chat-body {
    font-size: 0.86rem;
    color: #64748b;
    max-width: 340px;
    text-align: center;
    line-height: 1.55;
    margin: 0;
}

/* Source header */
.src-lbl {
    font-size: 0.69rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #2563eb;
    margin: 14px 0 6px;
}

/* Auth page */
.auth-hero {
    text-align: center;
    padding: 36px 0 20px;
}
.auth-hero-icon { font-size: 2.5rem; margin-bottom: 10px; }
.auth-hero-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0b1120;
    letter-spacing: -0.03em;
    margin: 0;
}
.auth-hero-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin: 6px 0 0;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def files_dir(username: str) -> Path:
    d = FILES_ROOT / username / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logout():
    for k in ["user", "selected_files", "messages"]:
        st.session_state.pop(k, None)


def file_icon(name: str) -> str:
    return {"pdf": "📄", "txt": "📝", "md": "📋"}.get(
        Path(name).suffix.lower().lstrip("."), "📎"
    )


def fmt_date(s: str) -> str:
    try:
        return datetime.fromisoformat(s).strftime("%b %d")
    except Exception:
        return s[:10] if len(s) >= 10 else s


# ─────────────────────────────────────────────────────────────────
# Auth page
# ─────────────────────────────────────────────────────────────────

def render_auth():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("""
        <div class="auth-hero">
            <div class="auth-hero-icon">⚡</div>
            <h1 class="auth-hero-title">SemanticSearch</h1>
            <p class="auth-hero-sub">AI-powered document search &amp; Q&amp;A</p>
        </div>
        """, unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["  Log in  ", "  Create account  "])

        with tab_in:
            st.write("")
            u = st.text_input("Username", key="li_u", placeholder="your-username")
            p = st.text_input("Password", type="password", key="li_p", placeholder="••••••••")
            st.write("")
            if st.button("Log in", type="primary", key="btn_li", use_container_width=True):
                # Bug fix #4: validate before sending to auth
                if not u.strip() or not p:
                    st.error("Please enter both username and password.")
                else:
                    ok, msg = login(u.strip(), p)
                    if ok:
                        st.session_state.user = msg
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_up:
            st.write("")
            u = st.text_input("Choose a username", key="su_u", placeholder="3+ characters")
            p = st.text_input("Choose a password", type="password", key="su_p",
                              placeholder="6+ characters")
            st.write("")
            if st.button("Create account", type="primary", key="btn_su",
                         use_container_width=True):
                # Bug fix #4: validate before sending to auth
                if not u.strip() or not p:
                    st.error("Please fill in both fields.")
                else:
                    ok, msg = signup(u.strip(), p)
                    if ok:
                        st.success(f"✅ {msg} — you can now log in.")
                    else:
                        st.error(msg)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

def render_sidebar(user: str) -> tuple[list[int], str]:
    """Renders the full sidebar; returns (selected_file_ids, mode)."""

    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────
        st.markdown("""
        <div class="sbar-logo">
            <div class="sbar-logo-mark">⚡</div>
            <span class="sbar-logo-text">SemanticSearch</span>
            <span class="sbar-logo-tag">BETA</span>
        </div>
        """, unsafe_allow_html=True)

        # ── User badge ─────────────────────────────────────────
        st.markdown(f"""
        <div class="sbar-user">
            <div class="sbar-avatar">{user[0].upper()}</div>
            <div>
                <div class="sbar-uname">{user}</div>
                <div class="sbar-ustatus">● active session</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Upload ─────────────────────────────────────────────
        st.markdown("<div class='sbar-section'>Upload documents</div>",
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "upload",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded:
            label = f"Index {len(uploaded)} file{'s' if len(uploaded) > 1 else ''}"
            if st.button(label, type="primary", use_container_width=True):
                root = files_dir(user)
                prog = st.progress(0, text="Starting…")
                results: list[tuple[str, int]] = []

                for idx, uf in enumerate(uploaded):
                    prog.progress(idx / len(uploaded),
                                  text=f"Processing {uf.name}…")
                    dest = root / uf.name
                    dest.write_bytes(uf.getbuffer())

                    # ── Bug fix #1: clear stale vectors before re-indexing ──
                    file_id = add_file(user, uf.name, str(dest), 0)
                    remove_file(user, file_id)            # removes old chunks/vectors
                    n = index_file(user, file_id, uf.name, str(dest))
                    add_file(user, uf.name, str(dest), n) # update chunk count

                    results.append((uf.name, n))

                prog.progress(1.0, text="Done!")
                prog.empty()
                for fname, n in results:
                    st.success(f"✅ {fname} — {n} chunks")
                st.rerun()

        st.divider()

        # ── File library ───────────────────────────────────────
        st.markdown("<div class='sbar-section'>Library</div>",
                    unsafe_allow_html=True)

        files = list_files(user)
        selected_ids: list[int] = list(st.session_state.get("selected_files", []))

        if not files:
            st.markdown("<div class='sbar-empty'>No documents yet — upload above.</div>",
                        unsafe_allow_html=True)
        else:
            # Stats row
            c1, c2 = st.columns(2)
            c1.metric("Files", len(files))
            c2.metric("Chunks", sum(f["n_chunks"] for f in files))

            # Bulk select controls
            ca, cb = st.columns(2)
            if ca.button("Select all", use_container_width=True):
                st.session_state.selected_files = [f["id"] for f in files]
                st.rerun()
            if cb.button("Clear", use_container_width=True):
                st.session_state.selected_files = []
                st.rerun()

            st.write("")
            new_selected: list[int] = []
            for f in files:
                icon = file_icon(f["filename"])
                col_cb, col_del = st.columns([10, 1])

                with col_cb:
                    checked = st.checkbox(
                        f"{icon}  {f['filename']}",
                        value=f["id"] in selected_ids,
                        key=f"cb_{f['id']}",
                        help=f"{f['n_chunks']} chunks · {fmt_date(f['uploaded_at'])}",
                    )
                    if checked:
                        new_selected.append(f["id"])

                with col_del:
                    if st.button("✕", key=f"del_{f['id']}", help="Remove file"):
                        delete_file(user, f["id"])
                        Path(f["path"]).unlink(missing_ok=True)
                        remove_file(user, f["id"])
                        st.rerun()

            st.session_state.selected_files = new_selected

        st.divider()

        # ── Query mode ─────────────────────────────────────────
        st.markdown("<div class='sbar-section'>Query mode</div>",
                    unsafe_allow_html=True)

        mode = st.radio(
            "mode",
            options=["🔍 Search only", "🧠 LLM only", "⚡ RAG (Search + LLM)"],
            index=2,
            label_visibility="collapsed",
            help=(
                "Search only: returns the top-k matching chunks.\n\n"
                "LLM only: answers from model knowledge — no document lookup.\n\n"
                "RAG: retrieves relevant chunks, then generates a grounded answer."
            ),
        )

        st.divider()

        # ── Utility buttons ────────────────────────────────────
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.write("")
        st.button("Log out", on_click=logout, use_container_width=True)

    return st.session_state.get("selected_files", []), mode


# ─────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────

def render_chat(user: str, selected_files: list[int], mode: str):
    use_search = "Search" in mode
    use_llm    = "LLM" in mode or "RAG" in mode

    mode_labels = {
        "🔍 Search only":        "🔍 Search",
        "🧠 LLM only":           "🧠 LLM",
        "⚡ RAG (Search + LLM)": "⚡ RAG",
    }
    mode_chip = mode_labels.get(mode, mode)
    n_sel = len(selected_files)

    # ── Header card ────────────────────────────────────────────
    st.markdown(f"""
    <div class="main-hdr">
        <div>
            <p class="main-hdr-title">Document Q&amp;A</p>
            <p class="main-hdr-sub">Search and query across your uploaded files</p>
        </div>
        <div>
            <div class="mode-chip">{mode_chip}</div>
            <div class="files-count">{n_sel} file{'s' if n_sel != 1 else ''} active</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Guard ──────────────────────────────────────────────────
    if use_search and not selected_files:
        st.info("Select at least one file in the sidebar to enable retrieval.")

    # ── Empty state ────────────────────────────────────────────
    st.session_state.setdefault("messages", [])

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-chat">
            <div class="empty-chat-icon">💬</div>
            <p class="empty-chat-title">Ready when you are</p>
            <p class="empty-chat-body">
                Upload documents in the sidebar, select the files you want to query,
                then type your question below.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.markdown("<div class='src-lbl'>Retrieved sources</div>",
                            unsafe_allow_html=True)
                for i, h in enumerate(msg["sources"], 1):
                    with st.expander(f"{i}. {h['filename']}  ·  score {h['score']:.3f}"):
                        st.code(h["text"], language=None)

    # ── Chat input ─────────────────────────────────────────────
    q = st.chat_input("Ask about your documents…")
    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        hits: list[dict] = []
        answer = ""

        # Retrieval
        if use_search:
            with st.spinner("Searching documents…"):
                hits = retrieve(user, q, file_ids=selected_files, k=5)

            if hits:
                st.markdown("<div class='src-lbl'>Retrieved sources</div>",
                            unsafe_allow_html=True)
                for i, h in enumerate(hits, 1):
                    with st.expander(f"{i}. {h['filename']}  ·  score {h['score']:.3f}"):
                        st.code(h["text"], language=None)
            else:
                st.caption("No relevant chunks found in the selected files.")

        # Generation
        if use_llm:
            with st.spinner("Generating answer…"):
                if use_search and hits:
                    context = build_context(hits)
                elif use_search:
                    context = "(no relevant context found in selected files)"
                else:
                    context = "(no context — answering from model knowledge)"
                answer = generate(q, context)

            st.markdown("**Answer**")
            st.write(answer)

        elif use_search:
            answer = "(LLM disabled — retrieval results shown above)"
            st.caption(answer)

        if not answer and not hits:
            answer = "No results. Try selecting files or switching mode."
            st.info(answer)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer or "(no answer)",
        "sources": hits,
    })


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    user = st.session_state.get("user")
    if not user:
        render_auth()
        return
    selected, mode = render_sidebar(user)
    render_chat(user, selected, mode)


if __name__ == "__main__":
    main()
