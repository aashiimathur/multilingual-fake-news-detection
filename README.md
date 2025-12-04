
Multilingual RAG + Cross-Encoder + NLI Stance Fact-Checking System

This project is an advanced **AI-powered fact‑checking pipeline** that combines:
- **Multilingual Retrieval-Augmented Generation (RAG)**
- **FAISS vector search**
- **Cross‑Encoder re‑ranking**
- **NLI stance classification (Support / Contradict / Neutral)**
- **Automatic multilingual input & output handling**
- **PDF ingestion**
- **Frontend built with Streamlit**
- **Backend built with FastAPI**

The system allows a user to input a claim (in *any Indian or global language*) and returns:
1. A final **TRUE / FAKE / MIXED** verdict  
2. A confidence score  
3. Translated reasoning in the original language  
4. Top retrieved evidence shown in 3 evidence cards  

---

🔥 Full System Flow (End‑to‑End)

**1. User Input**
The user enters a claim in **any language** (English, Hindi, Marathi, Tamil, Gujarati, Telugu, etc.)

Example:  
"सर्जिकल मास्क पहनने का केवल एक ही तरीका है।"

The frontend sends this to the FastAPI backend.

---

**2. Language Detection & Translation (Input)**
We detect the language automatically and translate the claim to **English** for model uniformity.

---

**3. Retrieval Using FAISS + Embeddings**
The English claim is embedded using:
- **sentence-transformers/all-mpnet-base-v2**

FAISS retrieves the top similar fact‑check statements from your dataset.

---

**4. Cross‑Encoder Re‑Ranking**
The top retrieved evidence is refined using:

**Model:**  
`cross-encoder/ms-marco-MiniLM-L-6-v2`

It scores how relevant each evidence is to the English claim.

---

**5. NLI Stance Classification**
Each evidence sentence is passed through an NLI classification model:

**Model:**  
`cross-encoder/nli-deberta-v3-base`

This determines:
- **Support** → Evidence supports the claim  
- **Contradict** → Evidence disproves the claim  
- **Neutral** → Irrelevant  

---

**6. Verdict Aggregation**
Based on stance probabilities:
- If mostly **support** → TRUE  
- If mostly **contradict** → FAKE  
- If mixed → PARTIALLY TRUE  
- If evidence confidence is low → USE ML‑ONLY fallback mode  

---

**7. Translation Back to User Language**
All outputs are translated:
- Reason  
- Evidence  
- Verdict explanation  

Using **googletrans**.

---

**8. Streamlit Frontend Presentation**
The results are displayed beautifully:
- Verdict badge  
- Reasoning section  
- 3 evidence cards **side‑by‑side**  
- Confidence meter  

---

🧠 Models Used

| Task | Model | Source |
|------|--------|---------|
| Embeddings | all-mpnet-base-v2 | SentenceTransformers |
| FAISS Indexing | IndexFlatL2 | Facebook AI / FAISS |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | HuggingFace |
| Stance Classification | cross-encoder/nli-deberta-v3-base | HuggingFace |
| Language Translation | googletrans 4.0 | Google Translate API |
| PDF Parsing | pdfplumber | Python |

---

📁 Project Structure

```
project/
│── backend/
│   ├── app.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── stance_ml.py
│   ├── translate.py
│   ├── utils.py
│── data/
│   ├── fact_base_clean.parquet
│   ├── fact_embeddings.npy
│   ├── faiss_index.bin
│── streamlit_app.py
│── README.md
```

---

🚀 How to Run

**1. Start Backend**
```bash
uvicorn backend.app:app --reload --port 8000
```

**2. Start Frontend**
```bash
streamlit run streamlit_app.py
```

---

🏁 Summary

This system is one of the most complete **multilingual fact‑checking AI pipelines**, offering:
✔ Multilingual claim support  
✔ Accurate retrieval using FAISS  
✔ State‑of‑the‑art re‑ranking  
✔ NLI stance inference  
✔ Beautiful Streamlit UI  
✔ Fully modular backend  


