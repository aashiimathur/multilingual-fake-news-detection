# backend/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_ce = None

def _get_ce():
    global _ce
    if _ce is None:
        # Load ONCE and correctly assign to _ce
        _ce = CrossEncoder(
            CROSS_ENCODER_MODEL,
            device="cpu"    # Fix meta-tensor GPU error
        )
    return _ce


def _extract_text(c: Dict) -> str:
    """
    Your app populates summary_en as the correct summary.
    So we use the same exact fields to avoid empty-text reranking.
    """
    return (
        c.get("summary_en")
        or c.get("summary")
        or c.get("full_text")
        or c.get("fact_text")
        or c.get("claim")
        or c.get("text")
        or ""
    )[:1024]


def rerank_with_cross_encoder(query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
    ce = _get_ce()

    if not candidates:
        return []

    # Build query-text pairs
    inputs = [[query, _extract_text(c)] for c in candidates]

    # Rerank scores
    scores = ce.predict(inputs)

    # Attach scores
    for c, s in zip(candidates, scores):
        c["_rerank_score"] = float(s)

    # Sort by rerank score
    ranked = sorted(
        candidates,
        key=lambda x: (x.get("_rerank_score", 0.0), x.get("score", 0.0)),
        reverse=True
    )

    return ranked[:top_n]