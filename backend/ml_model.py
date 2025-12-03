# ml_model.py

import joblib
import os

# -------------------------
# Load Model & Vectorizer
# -------------------------

MODEL_PATH = "models/fake_real_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("ML Pipeline Loaded Successfully!")
else:
    clf = None
    vectorizer = None
    print("⚠ ML model not found. Using dummy model.")


# -------------------------
# Predict Function
# -------------------------

def predict_fake_or_real(text: str):
    """
    Returns:
        verdict: "FAKE" or "REAL"
        reason: short explanation
    """

    if clf is None or vectorizer is None:
        # fallback
        return "UNSURE", "ML model not loaded."

    # Transform input
    X = vectorizer.transform([text])

    # Predict label
    pred = clf.predict(X)[0]

    # Confidence
    probs = clf.predict_proba(X)[0]
    confidence = max(probs)

    # Human-readable reason
    if pred == 1:
        verdict = "REAL"
        reason = f"Model predicts REAL with {confidence*100:.1f}% confidence."
    else:
        verdict = "FAKE"
        reason = f"Model predicts FAKE with {confidence*100:.1f}% confidence."

    return verdict, reason
