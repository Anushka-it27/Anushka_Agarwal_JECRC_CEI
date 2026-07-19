"""
hallucination_guard.py

Checks whether a generated answer is actually supported by the retrieved
context, using a Natural Language Inference (NLI) model as a fact-checker.

Approach:
  - Split the generated answer into individual sentences/claims.
  - For each claim, treat the retrieved context as the "premise" and the
    claim as the "hypothesis".
  - Run it through an NLI cross-encoder. If the model's top label is
    "contradiction" or the "entailment" score is below a threshold, flag
    the claim as unsupported (a likely hallucination or fabricated citation).

Model: cross-encoder/nli-deberta-v3-base (via sentence-transformers)
This is small enough to run on CPU and doesn't require an API key.
"""

import re
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_ENTAILMENT_THRESHOLD = 0.5

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def _split_into_claims(answer_text: str):
    """Naive sentence splitter — good enough for short RAG answers."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    return [s for s in sentences if len(s.strip()) > 0]


def check_answer(answer_text: str, context_chunks):
    """
    context_chunks: either a single string (legacy) or a list of individual
    context chunk strings. Checking against each chunk separately (rather than
    one giant merged block) avoids diluting the NLI model's signal - this
    model was trained on short sentence-pairs, not long multi-paragraph text,
    so feeding it one huge merged context tends to under-score everything as
    "neutral" even when a claim is well supported by one specific chunk.

    Returns a dict:
      {
        "claims": [
          {"claim": str, "label": "entailment"|"neutral"|"contradiction",
           "score": float, "flagged": bool, "best_chunk_index": int}
        ],
        "any_flagged": bool,
        "overall_support_ratio": float  # fraction of claims supported
      }
    """
    model = _get_model()
    claims = _split_into_claims(answer_text)
    if not claims:
        return {"claims": [], "any_flagged": False, "overall_support_ratio": 1.0}

    if isinstance(context_chunks, str):
        chunks = [context_chunks]
    else:
        chunks = list(context_chunks)
    if not chunks:
        chunks = [""]

    labels = ["contradiction", "entailment", "neutral"]

    results = []
    supported_count = 0
    for claim in claims:
        pairs = [(chunk, claim) for chunk in chunks]
        scores = model.predict(pairs, apply_softmax=True)  # shape (num_chunks, 3), each row sums to 1

        entailment_scores = scores[:, 1]
        best_idx = int(entailment_scores.argmax())
        best_entailment_score = float(entailment_scores[best_idx])
        top_label_idx = int(scores[best_idx].argmax())
        top_label = labels[top_label_idx]

        flagged = top_label == "contradiction" or best_entailment_score < _ENTAILMENT_THRESHOLD
        if not flagged:
            supported_count += 1

        results.append({
            "claim": claim,
            "label": top_label if best_entailment_score >= _ENTAILMENT_THRESHOLD else "neutral",
            "score": best_entailment_score,
            "flagged": flagged,
            "best_chunk_index": best_idx,
        })

    return {
        "claims": results,
        "any_flagged": any(r["flagged"] for r in results),
        "overall_support_ratio": supported_count / len(claims),
    }


if __name__ == "__main__":
    # Quick manual smoke test
    context = (
        "PR #4356 by tiangolo: Refactored dependency injection to use "
        "a cache to avoid recomputing dependencies with the same parameters "
        "within a single request."
    )
    answer = (
        "Dependency injection was refactored to cache results within a request. "
        "This change was authored by a Netflix engineer in 2019."
    )
    result = check_answer(answer, context)
    for c in result["claims"]:
        print(f"[{c['label']:>13} | {c['score']:.2f} | flagged={c['flagged']}] {c['claim']}")
