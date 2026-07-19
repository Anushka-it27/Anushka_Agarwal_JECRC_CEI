"""
run_ragas_eval.py

Runs the 50-question benchmark through the PatchContext pipeline and scores
the results with RAGAs metrics:
  - faithfulness: is the answer grounded in the retrieved context?
  - answer_relevancy: does the answer actually address the question?
  - context_precision: are the retrieved chunks relevant?
  - context_recall: (skipped by default — requires ground-truth answers,
    which this benchmark doesn't include; add "ground_truth" fields to
    benchmark_questions.json if you want this metric too)

Output: eval/ragas_results.csv and a printed summary.
"""

import sys
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qa_chain import PatchContextQA  # noqa: E402

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "ragas_results.csv"


def run():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = [q["question"] for q in json.load(f)]

    qa = PatchContextQA()

    records = {"question": [], "answer": [], "contexts": []}

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        docs = qa.retriever.invoke(question)
        context_texts = [d.page_content for d in docs]

        result = qa.ask(question)

        records["question"].append(question)
        records["answer"].append(result["answer"])
        records["contexts"].append(context_texts)

    dataset = Dataset.from_dict(records)

    print("\nRunning RAGAs evaluation (this calls the OpenAI API for scoring)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    df = scores.to_pandas()
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved detailed results to {RESULTS_PATH}")

    print("\n=== Summary (mean across 50 questions) ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric in df.columns:
            print(f"{metric:>20}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    run()
