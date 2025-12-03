import pandas as pd
import re
from tqdm import tqdm

DATA_PATH = "C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/factdrill_data.parquet"
df = pd.read_parquet(DATA_PATH)

def clean_text(t):
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t.strip())

def extract_verdict(text):
    """Infer verdict from investigation text."""
    t = text.lower()

    if any(k in t for k in ["false", "fake", "misleading", "incorrect", "wrong", "fabricated"]):
        return "FAKE"
    if any(k in t for k in ["true", "correct", "accurate", "not false"]):
        return "REAL"
    if any(k in t for k in ["partially", "mixed", "half true"]):
        return "PARTLY TRUE"
    return "UNVERIFIED"

def extract_summary(text):
    """Short summary of investigation."""
    if not isinstance(text, str):
        return ""
    # Take first 2 sentences
    sentences = re.split(r"[.!?]", text)
    summary = ". ".join(sentences[:2]).strip()
    return summary

clean_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    claim_text = clean_text(row["claim"])
    invest = clean_text(row["investigation"])

    verdict = extract_verdict(invest)
    summary = extract_summary(invest)

    clean_rows.append({
        "claim": claim_text,
        "summary": summary,
        "verdict": verdict,
        "source": row["link"],
        "date": row["publish_date"],
        "full_text": clean_text(row["document_text"])
    })

clean_df = pd.DataFrame(clean_rows)

clean_df.to_parquet("fact_base_clean.parquet")
clean_df.to_csv("fact_base_clean.csv", index=False)

print("\n🎉 Done! Clean fact base created:")
print(clean_df.head())
print("\nLabel counts:", clean_df["verdict"].value_counts())
