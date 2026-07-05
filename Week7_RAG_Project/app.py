import streamlit as st
from vectorstore import (
    extract_text_from_pdf,
    split_text_into_chunks,
    create_embeddings,
    store_in_pinecone,
    retrieve_relevant_chunks
)
from chatbot import generate_answer

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG-Based Document Question Answering")
st.markdown("Upload a PDF and ask questions based on its content.")

# ── Sidebar — API Keys ────────────────────────────────────
st.sidebar.header("🔑 API Configuration")

cohere_api_key = st.sidebar.text_input(
    "Cohere API Key",
    type="password",
    placeholder="Enter your Cohere API key"
)
pinecone_api_key = st.sidebar.text_input(
    "Pinecone API Key",
    type="password",
    placeholder="Enter your Pinecone API key"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Get your free API keys:**")
st.sidebar.markdown("- [Cohere](https://cohere.com)")
st.sidebar.markdown("- [Pinecone](https://pinecone.io)")

# ── Main Area ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file and cohere_api_key and pinecone_api_key:
        if st.button("🚀 Process Document"):
            with st.spinner("Processing PDF..."):
                try:
                    # Step 1: Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    st.success(f"✅ Text extracted successfully")

                    # Step 2: Chunk text
                    chunks = split_text_into_chunks(text)
                    st.success(f"✅ Created {len(chunks)} text chunks")

                    # Step 3: Create embeddings
                    embeddings = create_embeddings(chunks, cohere_api_key)
                    st.success(f"✅ Embeddings generated")

                    # Step 4: Store in Pinecone
                    store_in_pinecone(chunks, embeddings, pinecone_api_key)
                    st.success(f"✅ Stored in Pinecone vector database")

                    # Save in session state
                    st.session_state["processed"] = True
                    st.session_state["chunks"] = chunks
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    elif uploaded_file and (not cohere_api_key or not pinecone_api_key):
        st.warning("⚠️ Please enter both API keys in the sidebar first.")

with col2:
    st.subheader("💬 Ask a Question")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    query = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?"
    )

    if st.button("🔍 Get Answer"):
        if not query:
            st.warning("⚠️ Please enter a question.")
        elif not cohere_api_key or not pinecone_api_key:
            st.warning("⚠️ Please enter both API keys in the sidebar.")
        else:
            with st.spinner("Finding answer..."):
                try:
                    # Retrieve relevant chunks
                    context = retrieve_relevant_chunks(
                        query, cohere_api_key, pinecone_api_key
                    )

                    # Generate answer
                    answer = generate_answer(query, context, cohere_api_key)

                    # Save to chat history
                    st.session_state["chat_history"].append({
                        "question": query,
                        "answer": answer,
                        "context": context
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display chat history
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.subheader("📝 Chat History")
        for i, chat in enumerate(reversed(st.session_state["chat_history"])):
            with st.expander(f"Q: {chat['question']}", expanded=(i == 0)):
                st.markdown(f"**Answer:** {chat['answer']}")
                st.markdown("**Source Chunks Used:**")
                for j, chunk in enumerate(chat["context"]):
                    st.text_area(
                        f"Chunk {j+1}",
                        chunk,
                        height=80,
                        key=f"chunk_{i}_{j}"
                    )