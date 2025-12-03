# backend/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict
import os

# Use a lightweight cross-encoder for reranking (good balance of speed & quality)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_ce = None

def _get_ce():
    global _ce
    if _ce is None:
        _ce = CrossEncoder(CROSS_ENCODER_MODEL)
    return _ce


def rerank_with_cross_encoder(query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
    """
    candidates: list of dicts returned by retrieve_top_facts (score, claim, summary, ...)
    returns: same structure but reranked by cross-encoder score (descending)
    """
    ce = _get_ce()

    texts = []
    for c in candidates:
        # use summary or full_text snippet as candidate text
        text = c.get("summary") or c.get("full_text") or c.get("claim") or ""
        texts.append(text[:1024])  # limit length

    # compute scores
    inputs = [[query, t] for t in texts]
    rerank_scores = ce.predict(inputs)  # higher = better

    for c, s in zip(candidates, rerank_scores):
        c["_rerank_score"] = float(s)

    # sort descending by rerank score then by original sim
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x.get("_rerank_score", 0.0), x.get("score", 0.0)),
        reverse=True
    )

    return candidates_sorted[:top_n]
