import cohere

def generate_answer(query, context_chunks, cohere_api_key):
    """Generate answer using Cohere Chat API with retrieved context"""
    co = cohere.ClientV2(cohere_api_key)  # Use ClientV2 for new API

    # Combine context chunks
    context = "\n\n".join(context_chunks)

    # Build message
    messages = [
        {
            "role": "user",
            "content": f"""You are a helpful assistant that answers questions 
based on the provided document context.

Context from document:
{context}

User Question: {query}

Instructions:
- Answer only based on the context provided above
- If the answer is not in the context, say "I couldn't find 
  relevant information in the document."
- Be clear, concise, and accurate

Answer:"""
        }
    ]

    response = co.chat(
        model="command-r-plus-08-2024",
        messages=messages
    )

    return response.message.content[0].text.strip()