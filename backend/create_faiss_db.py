from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os

# ---- 1. Load or create dataset ----
DATA_PATH = "data/fact_dataset.csv"
os.makedirs("data", exist_ok=True)

# Example fallback data (you can replace this later with FactDrill)
if not os.path.exists(DATA_PATH):
    df = pd.DataFrame({
        "claim": [
            "COVID-19 vaccines are effective in preventing severe illness.",
            "The earth is flat.",
            "5G networks cause coronavirus."
        ],
        "label": ["TRUE", "FALSE", "FALSE"]
    })
    df.to_csv(DATA_PATH, index=False)

# ---- 2. Load data ----
df = pd.read_csv(DATA_PATH)
sentences = df["claim"].tolist()

# ---- 3. Create embeddings ----
print("Encoding sentences...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)

# ---- 4. Create FAISS index ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---- 5. Save index and model info ----
faiss.write_index(index, "data/fact_index.faiss")
np.save("data/fact_embeddings.npy", embeddings)
df.to_csv("data/fact_claims.csv", index=False)

print("✅ FAISS database created and saved in /data")
