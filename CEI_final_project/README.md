# PatchContext — RAG Pipeline over the FastAPI Repository

PatchContext is a Retrieval-Augmented Generation (RAG) pipeline built over the
[FastAPI repository](https://github.com/fastapi/fastapi)'s commit history,
pull requests, and issue threads. It lets engineers ask questions like
*"why was this designed this way?"* and get answers grounded in real
developer discussions, with clickable citations to commit SHAs, PR numbers,
and issue IDs.

## Stack

- **LangChain** — orchestration
- **FAISS** — vector store
- **OpenAI `text-embedding-ada-002`** — embeddings
- **OpenAI `gpt-4o-mini`** — answer generation
- **MMR (Maximal Marginal Relevance)** — retrieval, for diverse (non-redundant) results
- **NLI-based hallucination guard** (`cross-encoder/nli-deberta-v3-base`) — flags
  claims not supported by retrieved context
- **Streamlit** — UI
- **RAGAs** — evaluation on a 50-question benchmark

## Project structure

```
PatchContext/
├── app.py                       # Streamlit UI
├── requirements.txt
├── .env.example                 # copy to .env and fill in your keys
├── data/                        # raw ingested commits/PRs/issues (JSON)
├── faiss_index/                 # generated vector store (after build_index.py)
├── src/
│   ├── data_ingestion.py        # pulls commits/PRs/issues from GitHub API
│   ├── build_index.py           # chunks + embeds + saves FAISS index
│   ├── qa_chain.py               # MMR retrieval + gpt-4o-mini + citations
│   └── hallucination_guard.py   # NLI-based claim verification
└── eval/
    ├── benchmark_questions.json # 50-question benchmark
    └── run_ragas_eval.py        # scores the pipeline with RAGAs
```

## Setup

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Set up your API keys**
   ```
   copy .env.example .env
   ```
   Then edit `.env` and add your `OPENAI_API_KEY`. A `GITHUB_TOKEN` is
   optional but recommended (raises GitHub's rate limit from 60 to 5000
   requests/hour).

3. **Ingest data from the FastAPI repo**
   ```
   python src/data_ingestion.py
   ```
   This saves `data/commits.json`, `data/pull_requests.json`, and
   `data/issues.json`.

4. **Build the FAISS index**
   ```
   python src/build_index.py
   ```
   This embeds all ingested text and saves the index to `faiss_index/`.
   Requires your OpenAI API key to be set (this step costs a small amount
   of OpenAI credit for embeddings — typically a few cents for a project
   this size).

5. **Run the app**
   ```
   streamlit run app.py
   ```
   Enter your OpenAI API key in the sidebar and start asking questions.

6. **(Optional) Run the RAGAs evaluation**
   ```
   python eval/run_ragas_eval.py
   ```
   This runs all 50 benchmark questions through the pipeline and scores
   faithfulness, answer relevancy, and context precision. Results are saved
   to `eval/ragas_results.csv`. Note: this calls the OpenAI API repeatedly
   and will use some credits (still well under $1 total for this project).

## How the hallucination guard works

After `gpt-4o-mini` generates an answer, PatchContext splits it into
individual sentences ("claims") and checks each one against the retrieved
context using an NLI (Natural Language Inference) model. If a claim isn't
entailed by the context — or is directly contradicted by it — it's flagged
in the UI. This catches fabricated citations and unsupported claims before
they reach the user.

## Notes

- The retrieval uses MMR rather than plain similarity search, which
  reduces redundant/near-duplicate chunks in the context window and gives
  the LLM a broader, more diverse set of source material to draw from.
- Every chunk in the FAISS index carries citation metadata (commit SHA,
  PR number, or issue number) back to its source, so the QA chain can
  never lose track of where a claim came from.
