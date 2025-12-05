Multilingual Fake News Detection System - README

# Multilingual Fake News Detection System

## Overview
This project identifies misinformation across multiple languages using a hybrid pipeline of semantic retrieval, stance classification, and ML fallback models.

## Features
- Multilingual claim input support
- Translation to/from English
- Evidence retrieval with FAISS
- Cross-encoder reranking
- NLI stance classification
- ML fallback classifier
- Streamlit frontend + FastAPI backend

## Architecture
1. User submits claim (any language)
2. Language detection & translation
3. Semantic retrieval (MPNet + FAISS)
4. Reranking (Cross-Encoder)
5. Stance classification (NLI DeBERTa)
6. Verdict aggregation
7. ML fallback if needed
8. Translation back to user language
9. Render output in Streamlit

## Technologies
- FastAPI
- Streamlit
- FAISS
- HuggingFace Transformers
- SentenceTransformers
- GoogleTranslator (deep-translator)

## Setup
```
pip install -r requirements.txt
uvicorn backend.app:app --reload
streamlit run streamlit_app.py
```

## Dataset
FactDrill (2013–2020) — multilingual fact-checked dataset with 22k+ instances.

## Future Improvements
- Better multilingual embeddings
- Improved fallback classifier
- Larger fact-check datasets
- Native Indic-language NLI models

