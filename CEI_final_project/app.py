"""
app.py

Streamlit front-end for PatchContext.
Ask "why was this designed this way?" about the FastAPI repo and get answers
grounded in real commits, PRs, and issues, with clickable citations.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

st.set_page_config(page_title="PatchContext", page_icon="🔍", layout="wide")

# ---------- Sidebar: API key + index status ----------
with st.sidebar:
    st.title("⚙️ Settings")
    api_key_input = st.text_input("Cohere API Key", type="password")
    if api_key_input:
        os.environ["COHERE_API_KEY"] = api_key_input

    st.markdown("---")
    index_path = Path(__file__).resolve().parent / "faiss_index"
    if index_path.exists():
        st.success("FAISS index found ✅")
    else:
        st.error("No FAISS index found. Run `python src/build_index.py` first.")

    st.markdown("---")
    st.caption(
        "PatchContext is a RAG pipeline over the FastAPI repository's commit "
        "history, pull requests, and issue threads."
    )

st.title("🔍 PatchContext")
st.caption("RAG Pipeline over the FastAPI Repository")

st.markdown(
    "Ask a question about **why** something in FastAPI was designed the way "
    "it was, and get an answer grounded in real developer discussions — with "
    "clickable citations."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. Why was dependency injection caching introduced?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not os.environ.get("COHERE_API_KEY"):
        st.warning("Please enter your Cohere API key in the sidebar first.")
    elif not question.strip():
        st.warning("Please type a question.")
    elif not index_path.exists():
        st.warning("No index found. Run `python src/build_index.py` first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            from qa_chain import PatchContextQA

            qa = st.session_state.get("qa_instance")
            if qa is None:
                qa = PatchContextQA()
                st.session_state["qa_instance"] = qa

            result = qa.ask(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Citations")
        if result["citations"]:
            for c in result["citations"]:
                if c["url"]:
                    st.markdown(f"- [{c['label']}]({c['url']})")
                else:
                    st.markdown(f"- {c['label']}")
        else:
            st.write("No citations returned.")

        st.subheader("Hallucination Check")
        guard = result["hallucination_check"]
        support_pct = guard["overall_support_ratio"] * 100
        if guard["any_flagged"]:
            st.warning(f"⚠️ Some claims may not be fully supported by retrieved context ({support_pct:.0f}% supported).")
        else:
            st.success(f"✅ All claims appear supported by retrieved context ({support_pct:.0f}% supported).")

        with st.expander("See claim-by-claim breakdown"):
            for c in guard["claims"]:
                flag = "🚩" if c["flagged"] else "✔️"
                st.write(f"{flag} **[{c['label']}, {c['score']:.2f}]** {c['claim']}")

        st.caption(f"Retrieved {result['num_chunks_retrieved']} chunks via MMR.")
