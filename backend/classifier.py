# backend/classifier.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# --------------------------------------------
# LLM Classification (RAG)
# --------------------------------------------
def classify_claim_rag(claim, evidence):

    prompt = f"""
You are a fact-checking assistant. Use ONLY the evidence given.

Claim: "{claim}"
Evidence: "{evidence}"

Respond in JSON with keys:
{{
  "verdict": "TRUE / FALSE / UNVERIFIED",
  "reason": "...",
  "confidence": 0.0-1.0
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=400)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # --------- STEP 1: Try direct JSON ----------
    try:
        j = json.loads(extract_json(text))
        return j
    except:
        pass

    # --------- STEP 2: Try TOON format ----------
    try:
        j = toon_to_json(text)
        return j
    except:
        pass

    # --------- STEP 3: If LLM failed completely ----------
    return {
        "verdict": "UNVERIFIED",
        "reason": "Model output invalid.",
        "confidence": 0.1
    }


# ----------------------
# Extract JSON
# ----------------------
def extract_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else "{}"


# ----------------------
# TOON → JSON converter
# ----------------------
def toon_to_json(text):
    lines = text.split("\n")
    out = {}

    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            out[key.strip().lower()] = val.strip()

    return {
        "verdict": out.get("verdict", "UNVERIFIED").upper(),
        "reason": out.get("reason", "No explanation."),
        "confidence": float(out.get("confidence", 0.2))
    }
