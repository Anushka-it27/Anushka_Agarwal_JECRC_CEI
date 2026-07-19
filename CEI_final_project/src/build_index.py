"""
build_index.py

Turns the raw ingested commits/PRs/issues into a searchable FAISS vector store.

Each chunk keeps metadata pointing back to its source (commit SHA, PR number,
or issue number) so the QA chain can always produce a real citation instead
of a fabricated one.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)


def _load(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Warning: {path} not found. Run data_ingestion.py first.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def commits_to_documents(commits):
    docs = []
    for c in commits:
        text = f"Commit {c['short_sha']} by {c['author']} on {c['date']}:\n{c['message']}"
        for chunk in SPLITTER.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "commit",
                    "citation": c["short_sha"],
                    "url": c["url"],
                },
            ))
    return docs


def prs_to_documents(prs):
    docs = []
    for pr in prs:
        text = f"Pull Request #{pr['number']} by {pr['user']}: {pr['title']}\n\n{pr['body']}"
        for chunk in SPLITTER.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "pull_request",
                    "citation": f"PR #{pr['number']}",
                    "url": pr["url"],
                },
            ))
    return docs


def issues_to_documents(issues):
    docs = []
    for issue in issues:
        text = f"Issue #{issue['number']} by {issue['user']}: {issue['title']}\n\n{issue['body']}"
        for chunk in SPLITTER.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_type": "issue",
                    "citation": f"Issue #{issue['number']}",
                    "url": issue["url"],
                },
            ))
    return docs


def build_and_save_index():
    commits = _load("commits.json")
    prs = _load("pull_requests.json")
    issues = _load("issues.json")

    all_docs = (
        commits_to_documents(commits)
        + prs_to_documents(prs)
        + issues_to_documents(issues)
    )

    if not all_docs:
        raise RuntimeError(
            "No documents to index. Run src/data_ingestion.py first to populate data/."
        )

    print(f"Embedding {len(all_docs)} chunks with {EMBEDDING_MODEL_NAME} (local, free)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_and_save_index()
