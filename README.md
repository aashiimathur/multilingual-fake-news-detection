
# Multilingual Fake News Detection – README

## Overview
This project implements a multilingual fact-checking system using a hybrid RAG (Retrieval-Augmented Generation) pipeline combined with NLI stance detection and an ML fallback classifier. The system supports claims in multiple languages, retrieves relevant evidence from a FAISS-based fact database, classifies stance, aggregates final verdicts, and returns explainable results with evidence.

## Key Features
- Multilingual claim support (auto-detect + translation)
- FAISS-based semantic retrieval using mpnet embeddings
- Cross-encoder reranking for high-quality evidence selection
- NLI stance classification (support/contradict/neutral)
- ML fallback classifier using DeBERTa-v3 MNLI-FEVER-ANLI
- Explainable output with best evidence sentence
- Streamlit UI + FastAPI backend

## Architecture
1. Claim received from Streamlit.
2. Language detected → translated to English.
3. Embedding generated using all-mpnet-base-v2.
4. FAISS retrieves top semantic matches.
5. Cross-encoder reranks evidence.
6. NLI stance classifier selects best sentence + stance.
7. Verdict aggregated from stance results.
8. If evidence weak → fallback ML classifier.
9. Output translated back to user language.
10. Returned to UI.

## Technologies Used
- Python, FastAPI, Streamlit
- FAISS for vector search
- SentenceTransformers for embeddings & cross encoders
- DeBERTa-v3 for NLI fallback classifier
- GoogleTranslator for multilingual support
- pdfplumber for PDF ingestion

## File Structure
backend/
- app.py
- retrieval.py
- reranker.py
- stance_ml.py
- ml_fallback.py
- translate.py
- utils.py

frontend/
- streamlit_app.py

data/
- fact_base_clean.parquet
- fact_embeddings.npy
- faiss_index.bin

## Running the Project
1. Start FastAPI backend:
   uvicorn backend.app:app --reload --port 8000

2. Start Streamlit UI:
   streamlit run streamlit_app.py

## Dataset
FactDrill Dataset:
- 22,435 fact-checked social media claims across India
- 13 languages
- 2013–2020 span
- Cleaned summaries + extracted verdicts stored as fact_base_clean.parquet

## Retrieval Process (Theory)
- Text is converted into dense vectors (768-dim mpnet-base embeddings).
- Stored in FAISS index enabling fast nearest-neighbor search.
- FAISS computes L2 distances between vectors.
- Cross-encoder reranking refines top retrieved evidence.

## Stance Classification (Theory)
- Evidence split into sentences.
- Semantic similarity computed between claim & sentences.
- NLI model predicts stance per sentence.
- Confidence = similarity × NLI confidence.
- Best sentence selected as explanation.

## Verdict Aggregation Logic
- Majority support → TRUE
- Majority contradict → FAKE
- Mix/low confidence → UNVERIFIED
- Weak evidence → ML fallback (DeBERTa)

## ML Fallback
- Predicts TRUE / FAKE / UNVERIFIED using NLI probabilities.
- Ensures system always returns a verdict.

## Multilingual Support
- Langdetect identifies language.
- GoogleTranslator performs:
   - claim → English
   - results → user language

## UI Experience
- Verdict badge
- Confidence meter
- Evidence list
- Multilingual reasoning
- PDF upload support

## Future Improvements
- Better summarization model
- Larger embedding model
- Finetuned stance classifier
- Vectorstore sharding for scale
- Real-time dataset updates


