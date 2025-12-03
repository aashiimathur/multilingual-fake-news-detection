import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

MODEL_NAME = "intfloat/multilingual-e5-base"
EMB_PATH = os.path.join("C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/fact_embeddings.npy")
PARQUET_PATH = os.path.join("C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/fact_base_clean.parquet")
FAISS_PATH = os.path.join("C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/faiss_index.bin")

print("Loading multilingual embedder:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

print("Loading dataset:", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH)

# Choose text column properly
if "summary" in df.columns:
    texts = df["summary"].fillna("").astype(str).tolist()
elif "fact_text" in df.columns:
    texts = df["fact_text"].fillna("").astype(str).tolist()
elif "text" in df.columns:
    texts = df["text"].fillna("").astype(str).tolist()
else:
    raise ValueError("No usable text column in dataframe.")

print("Embedding", len(texts), "documents...")

embeddings = []
batch_size = 64

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=False)
    embeddings.append(emb)

embeddings = np.vstack(embeddings).astype("float32")
print("Embedding shape:", embeddings.shape)

np.save(EMB_PATH, embeddings)
print("Saved embeddings:", EMB_PATH)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_PATH)
print("Saved FAISS index:", FAISS_PATH)

print("DONE ✓")
