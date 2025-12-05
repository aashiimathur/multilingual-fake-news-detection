# backend/ml_fallback.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class MLFallbackClassifier:
    _tokenizer = None
    _model = None

    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.model_name = model_name
        self.THRESHOLD = 0.60   # Confidence threshold for TRUE/FAKE

    def _load_model(self):
        """Lazy-load + cache tokenizer/model"""
        if MLFallbackClassifier._model is None:
            print(f"⚡ Loading factuality model: {self.model_name}")
            MLFallbackClassifier._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            MLFallbackClassifier._model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            MLFallbackClassifier._model.eval()
        return self._tokenizer, self._model

    def predict(self, claim_en: str):
        """Return a clean fallback verdict independent of RAG."""
        tok, model = self._load_model()

        with torch.no_grad():
            inputs = tok(claim_en, return_tensors="pt", truncation=True, padding=True)
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0).tolist()

        # FEVER-style mapping
        contr, neutral, entail = probs

        # Choose winner with threshold logic
        if contr >= self.THRESHOLD:
            return {
                "fallback_pred": "FAKE",
                "fallback_confidence": round(contr * 100, 2),
                "probs": {"contr": contr, "neutral": neutral, "entail": entail}
            }

        if entail >= self.THRESHOLD:
            return {
                "fallback_pred": "TRUE",
                "fallback_confidence": round(entail * 100, 2),
                "probs": {"contr": contr, "neutral": neutral, "entail": entail}
            }

        # Otherwise weak → UNVERIFIED
        # confidence = max(probs)
        return {
            "fallback_pred": "UNVERIFIED",
            "fallback_confidence": round(max(probs) * 100, 2),
            "probs": {"contr": contr, "neutral": neutral, "entail": entail}
        }
    