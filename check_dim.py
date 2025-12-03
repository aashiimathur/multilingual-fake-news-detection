import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("data/faiss_index.bin")
print("🧠 FAISS index dimension:", index.d)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("🧩 Embedding dimension from current model:", model.get_sentence_embedding_dimension())
