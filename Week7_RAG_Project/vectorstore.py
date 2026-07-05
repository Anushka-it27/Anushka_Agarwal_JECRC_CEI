import cohere
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """Extract raw text from uploaded PDF"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text):
    """Split text into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return chunks

def create_embeddings(chunks, cohere_api_key):
    """Convert text chunks into vector embeddings using Cohere"""
    co = cohere.Client(cohere_api_key)
    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def store_in_pinecone(chunks, embeddings, pinecone_api_key, index_name="rag-index"):
    """Store embeddings in Pinecone vector database"""
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,   # Cohere embed-english-v3.0 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    # Upsert vectors with text as metadata
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    print(f"Stored {len(vectors)} chunks in Pinecone.")
    return index

def retrieve_relevant_chunks(query, cohere_api_key, pinecone_api_key,
                              index_name="rag-index", top_k=3):
    """Retrieve top-k relevant chunks for a user query"""
    co = cohere.Client(cohere_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    # Embed the query
    query_embedding = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Extract text from results
    context_chunks = [match["metadata"]["text"] for match in results["matches"]]
    return context_chunks