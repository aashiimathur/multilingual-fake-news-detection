# backend/stance_ml.py
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder
import torch
from backend.translate import translate_to_english

# Download sentence tokenizer if missing
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

NLI_MODEL = "cross-encoder/nli-deberta-base"
_nli = None

def get_nli():
    global _nli
    if _nli is None:
        _nli = CrossEncoder(NLI_MODEL)
    return _nli

LABEL_MAP = {0: "contradict", 1: "neutral", 2: "support"}


def classify_sentence_level(claim: str, evidence_text: str):
    """Split evidence into sentences and classify stance PER SENTENCE"""
    nli = get_nli()
    evidence_text = translate_to_english(evidence_text)
    sentences = sent_tokenize(evidence_text)

    pairs = [(claim, s) for s in sentences]
    probs = nli.predict(pairs, apply_softmax=True)

    results = []
    for s, p in zip(sentences, probs):
        label_id = int(torch.argmax(torch.tensor(p)))
        stance = LABEL_MAP[label_id]
        conf = float(max(p)) * 100
        results.append((s, stance, conf, p.tolist()))

    # pick strongest stance sentence
    best = max(results, key=lambda x: x[2])
    return best   # (sentence, stance, conf, raw_probs)


def classify_stance_ml(claim: str, evidence_list):
    """Apply sentence-level stance classification to each evidence item"""

    out = []
    for ev in evidence_list:
        text = ev.get("summary") or ""

        best_sentence, stance, conf, raw = classify_sentence_level(claim, text)

        enriched = dict(ev)
        enriched["best_sentence"] = best_sentence
        enriched["stance"] = stance
        enriched["stance_confidence"] = conf
        enriched["nli_raw"] = raw

        out.append(enriched)

    return out


def aggregate_ml_verdict(evidences):
    """Use aggregated stance to decide TRUE/FAKE/etc."""

    if not evidences:
        return "UNVERIFIED", 0.0

    support = [e for e in evidences if e["stance"] == "support"]
    contradict = [e for e in evidences if e["stance"] == "contradict"]

    # Compute best stance confidence
    best_conf = max(e["stance_confidence"] for e in evidences)

    # If too low → fallback to ML classifier pipeline
    if best_conf < 60:
        return "USE_ML_MODEL", best_conf

    # Majority vote
    if len(contradict) > len(support):
        return "FAKE", best_conf
    elif len(support) > len(contradict):
        return "TRUE", best_conf
    elif support and contradict:
        return "MISLEADING", best_conf

    return "UNVERIFIED", best_conf
