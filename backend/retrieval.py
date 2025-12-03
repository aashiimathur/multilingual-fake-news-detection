# backend/retrieval.py
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PATHS (YOUR ORIGINAL ONES)
# =========================
EMB_PATH = "C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/fact_embeddings.npy"
PARQUET_PATH = "C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/fact_base_clean.parquet"
FAISS_INDEX_PATH = "C:/Users/DELL/OneDrive/Documents/SEM 5/EDAI-5/edai5-rag/data/faiss_index.bin"

# =========================
# LOAD ORIGINAL EMBEDDINGS
# =========================
embeddings = np.load(EMB_PATH).astype("float32")
EMB_DIM = embeddings.shape[1]
NUM_DOCS = embeddings.shape[0]

# =========================
# LOAD TEXTS
# =========================
docs_df = pd.read_parquet(PARQUET_PATH)
if "summary" in docs_df.columns:
    DOC_TEXTS = docs_df["summary"].fillna("")
elif "text" in docs_df.columns:
    DOC_TEXTS = docs_df["text"].fillna("")
else:
    DOC_TEXTS = docs_df.astype(str).agg(" ".join, axis=1).fillna("")

# =========================
# LOAD FAISS INDEX
# =========================
if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatL2(EMB_DIM)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

# =========================
# ORIGINAL EMBEDDER (VERY IMPORTANT)
# same model used to create the embeddings!!
# =========================
_embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# =========================
# RETRIEVE FUNCTION
# =========================
def retrieve_top_facts(query: str, top_k: int = 5):
    if not query.strip():
        return []

    q_vec = _embedder.encode([query]).astype("float32")
    D, I = faiss_index.search(q_vec, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        text = DOC_TEXTS.iloc[idx] if idx < len(DOC_TEXTS) else ""
        results.append({
            "summary": str(text),
            "score": float(dist),
            "idx": int(idx)
        })
    return results
