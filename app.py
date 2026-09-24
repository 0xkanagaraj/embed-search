import streamlit as st
from pathlib import Path

from auth import signup, login
from db import add_file, list_files, delete_file
from pipeline import index_file, remove_file, retrieve, build_context
from llm import generate

st.set_page_config(page_title="Document Q&A", layout="wide")

FILES_ROOT = Path("data/users")


def files_dir(username: str) -> Path:
    d = FILES_ROOT / username / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logout():
    for k in ["user", "selected_files", "messages"]:
        st.session_state.pop(k, None)


def render_auth():
    st.title("📄 Document Q&A")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pw")
        if st.button("Log in", type="primary"):
            ok, msg = login(u, p)
            if ok:
                st.session_state.user = msg
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        u = st.text_input("Choose a username", key="su_user")
        p = st.text_input("Choose a password", type="password", key="su_pw")
        if st.button("Create account"):
            ok, msg = signup(u, p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


def render_upload(user: str):
    with st.expander("📤 Upload documents", expanded=False):
        uploaded = st.file_uploader(
            "PDF / TXT / MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Process uploads", type="primary"):
            root = files_dir(user)
            for uf in uploaded:
                dest = root / uf.name
                dest.write_bytes(uf.getbuffer())
                with st.spinner(f"Indexing {uf.name}…"):
                    file_id = add_file(user, uf.name, str(dest), 0)
                    n = index_file(user, file_id, uf.name, str(dest))
                    add_file(user, uf.name, str(dest), n)
                st.success(f"{uf.name}: {n} chunks")


def render_file_selector(user: str) -> list[int]:
    st.subheader("📚 Your files")
    files = list_files(user)
    if not files:
        st.info("No files yet. Upload some above.")
        return []

    selected = set(st.session_state.get("selected_files", []))

    c1, c2, c3 = st.columns([6, 1, 1])
    c1.caption(f"{len(files)} file(s) · {sum(f['n_chunks'] for f in files)} chunks")
    if c2.button("Select all"):
        st.session_state.selected_files = [f["id"] for f in files]
        st.rerun()
    if c3.button("Clear"):
        st.session_state.selected_files = []
        st.rerun()

    new_selected = []
    for f in files:
        col1, col2, col3 = st.columns([6, 1, 1])
        checked = col1.checkbox(
            f"**{f['filename']}** — {f['n_chunks']} chunks",
            value=f["id"] in selected,
            key=f"cb_{f['id']}",
        )
        if checked:
            new_selected.append(f["id"])
        col2.caption(f["uploaded_at"][:10])
        if col3.button("🗑", key=f"del_{f['id']}"):
            delete_file(user, f["id"])
            Path(f["path"]).unlink(missing_ok=True)
            remove_file(user, f["id"])
            st.rerun()

    st.session_state.selected_files = new_selected
    return new_selected


def render_qa(user: str, selected_files: list[int]):
    st.subheader("💬 Ask questions")

    # --- Mode selector ---
    mode = st.radio(
        "Mode",
        options=["🔍 Semantic search only", "🧠 LLM only", "⚡ RAG (search + LLM)"],
        index=2,
        horizontal=True,
        key="qa_mode",
        help=(
            "Semantic search only → returns the top-k chunks.\n\n"
            "LLM only → answers from model knowledge, not your docs.\n\n"
            "RAG → retrieves chunks, then asks the LLM to answer from them."
        ),
    )
    use_search = mode in ("🔍 Semantic search only", "⚡ RAG (search + LLM)")
    use_llm    = mode in ("🧠 LLM only", "⚡ RAG (search + LLM)")

    if use_search and not selected_files:
        st.info("Select at least one file to enable retrieval.")
        return

    st.session_state.setdefault("messages", [])

    # --- Render full history (sources included) ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.markdown(f"### 📑 Top-{len(msg['sources'])} retrieved chunks")
                for i, h in enumerate(msg["sources"], 1):
                    with st.expander(
                        f"{i}. [{h['score']:.3f}] {h['filename']}"
                    ):
                        st.code(h["text"], language=None)

    q = st.chat_input("Ask something…")
    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        hits = []
        answer = ""

        # ---- Retrieval (optional) ----
        if use_search:
            with st.spinner("Retrieving…"):
                hits = retrieve(user, q, file_ids=selected_files, k=5)

            if hits:
                st.markdown(f"### 📑 Top-{len(hits)} retrieved chunks")
                for i, h in enumerate(hits, 1):
                    with st.expander(
                        f"{i}. [{h['score']:.3f}] {h['filename']}"
                    ):
                        st.code(h["text"], language=None)
            else:
                st.caption("No relevant chunks found.")

        # ---- LLM (optional) ----
        if use_llm:
            with st.spinner("Generating answer…"):
                if use_search and hits:
                    context = build_context(hits)
                elif use_search and not hits:
                    context = "(no relevant context was found in the selected files)"
                else:
                    context = "(no context provided — answer from your own knowledge)"

                answer = generate(q, context)

            st.markdown("### 🧠 Answer")
            st.write(answer)

        elif use_search:
            answer = "(LLM disabled — showing retrieval results only)"
            st.caption(answer)

        # ---- Fallback if neither produced anything useful ----
        if not answer and not hits:
            answer = "Nothing to show — enable a mode above."
            st.write(answer)

    # ---- Persist EVERYTHING to history ----
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer or "(no answer)",
        "sources": hits,          # empty list if search was off
    })


def main():
    user = st.session_state.get("user")
    if not user:
        render_auth()
        return

    st.sidebar.title(f"👤 {user}")
    st.sidebar.button("Log out", on_click=logout)

    st.title("📄 Document Q&A")
    render_upload(user)
    selected = render_file_selector(user)
    st.divider()
    render_qa(user, selected)


if __name__ == "__main__":
    main()