# backend/stance_ml.py
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
from backend.translate import translate_to_english

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

NLI_MODEL = "cross-encoder/nli-deberta-base"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

_nli = None
_emb = None


def get_nli():
    global _nli
    if _nli is None:
        _nli = CrossEncoder(NLI_MODEL)
    return _nli


def get_emb():
    global _emb
    if _emb is None:
        _emb = SentenceTransformer(EMB_MODEL)
    return _emb


LABEL_MAP = {0: "contradict", 1: "neutral", 2: "support"}


# ---------------------------------------------------------
# Subject detection — fallback for vague claims
# ---------------------------------------------------------
def is_low_information_claim(claim: str) -> bool:
    tokens = word_tokenize(claim.lower())
    if len(tokens) < 3:
        return True

    meaningless = {"this", "that", "it", "true", "false", "real", "correct"}
    if all(token in meaningless for token in tokens):
        return True

    question_words = {"what", "who", "why", "when", "how", "is", "are"}
    if tokens[0] in question_words and len(tokens) < 4:
        return True

    return False


# ---------------------------------------------------------
# Sentence-level stance with semantic filtering
# ---------------------------------------------------------
def classify_sentence_level(claim: str, evidence_text: str):

    evidence_text = translate_to_english(evidence_text)
    sentences = sent_tokenize(evidence_text)

    if not sentences:
        return ("", "neutral", 0.0, [], 0.0, 0.0)

    nli = get_nli()
    emb = get_emb()

    claim_emb = emb.encode(claim, convert_to_tensor=True)
    sent_embs = emb.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, sent_embs)[0]

    pairs = [(claim, s) for s in sentences]
    probs = nli.predict(pairs, apply_softmax=True)

    results = []
    for s, sim, p in zip(sentences, sims, probs):
        label_id = int(torch.argmax(torch.tensor(p)))
        stance = LABEL_MAP[label_id]
        nli_conf = max(p) * 100
        combined_conf = float(sim.item()) * nli_conf

        results.append((s, stance, combined_conf, p.tolist(), float(sim), float(nli_conf)))

    best = max(results, key=lambda x: x[2])
    return best


# ---------------------------------------------------------
# Apply stance classifier to RAG evidence
# ---------------------------------------------------------
def classify_stance_ml(claim: str, evidence_list):

    # 🔥 rule: vague claims → skip stance and fallback
    if is_low_information_claim(claim):
        return []

    out = []
    for ev in evidence_list:
        text = ev.get("summary") or ""
        best_sentence, stance, combined_conf, raw_probs, sim, nli_conf = classify_sentence_level(claim, text)

        enriched = dict(ev)
        enriched["best_sentence"] = best_sentence
        enriched["stance"] = stance
        enriched["stance_confidence"] = combined_conf
        enriched["semantic_similarity"] = sim
        enriched["nli_raw"] = raw_probs
        enriched["nli_confidence"] = nli_conf

        out.append(enriched)

    return out


# ---------------------------------------------------------
# FINAL aggregation with strong fallback triggers
# ---------------------------------------------------------
def aggregate_ml_verdict(evidences):

    # empty OR vague → fallback
    if not evidences:
        return "USE_ML_MODEL", 0.0

    sims = [e["semantic_similarity"] for e in evidences]
    avg_sim = sum(sims) / len(sims)

    # irrelevant evidence → fallback
    if avg_sim < 0.18:
        return "USE_ML_MODEL", avg_sim * 100

    support = [e for e in evidences if e["stance"] == "support"]
    contradict = [e for e in evidences if e["stance"] == "contradict"]

    best_conf = max(e["stance_confidence"] for e in evidences)

    # NLI model unsure → fallback
    # fallback if strongest NLI probability < 60%
    best_nli_conf = max(e["nli_confidence"] for e in evidences)
    if best_nli_conf < 60:
        return "USE_ML_MODEL", best_nli_conf

    # normal classification
    if len(contradict) > len(support):
        return "FAKE", best_conf
    if len(support) > len(contradict):
        return "TRUE", best_conf

    return "UNVERIFIED", best_conf
