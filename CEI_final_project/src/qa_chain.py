"""
qa_chain.py

The core PatchContext pipeline:
  1. Retrieve relevant chunks from FAISS using MMR (Maximal Marginal Relevance)
     for diverse, non-redundant context.
  2. Generate an answer with gpt-4o-mini, instructed to cite sources inline.
  3. Run the answer through the NLI hallucination guard.
  4. Return the answer + clickable citations + a support/confidence signal.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from hallucination_guard import check_answer

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are PatchContext, an assistant that explains design decisions \
in the FastAPI codebase by grounding every answer in real commit messages, pull \
request discussions, and issue threads.

Rules:
- Only make claims that are directly supported by the provided context.
- Cite your sources inline using the citation tags given in the context \
  (e.g. "(PR #4356)" or "(commit a1b2c3d)" or "(Issue #1234)").
- If the context does not contain enough information to answer, say so plainly \
  instead of guessing.
- Keep answers concise and technical.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])


def _load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def _format_context(docs):
    blocks = []
    for d in docs:
        citation = d.metadata.get("citation", "unknown source")
        blocks.append(f"[{citation}] {d.page_content}")
    return "\n\n".join(blocks)


def _format_citations(docs):
    seen = set()
    citations = []
    for d in docs:
        citation = d.metadata.get("citation")
        url = d.metadata.get("url")
        if citation and citation not in seen:
            seen.add(citation)
            citations.append({"label": citation, "url": url})
    return citations


class PatchContextQA:
    def __init__(self, k=6, fetch_k=20, lambda_mult=0.5, model="command-r7b-12-2024"):
        """
        k: number of chunks to return after MMR re-ranking
        fetch_k: number of candidates MMR selects from before diversifying
        lambda_mult: 0 = max diversity, 1 = max relevance (0.5 is a balanced default)
        model: Cohere's free-tier-friendly "command-r" model
        """
        self.vectorstore = _load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
        self.llm = ChatCohere(model=model, temperature=0)

    def ask(self, question: str):
        docs = self.retriever.invoke(question)
        context = _format_context(docs)
        citations = _format_citations(docs)
        individual_chunks = [d.page_content for d in docs]

        chain = PROMPT | self.llm
        response = chain.invoke({"context": context, "question": question})
        answer_text = response.content

        guard_result = check_answer(answer_text, individual_chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "hallucination_check": guard_result,
            "num_chunks_retrieved": len(docs),
        }


if __name__ == "__main__":
    qa = PatchContextQA()
    result = qa.ask("Why was dependency injection caching introduced?")
    print(result["answer"])
    print("\nCitations:", result["citations"])
    print("\nSupport ratio:", result["hallucination_check"]["overall_support_ratio"])
