# backend/app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.retrieval import retrieve_top_facts
from backend.reranker import rerank_with_cross_encoder
from backend.utils import normalize_text, extract_text_from_pdf
from backend.stance_ml import classify_stance_ml, aggregate_ml_verdict
from backend.translate import (
    detect_lang,
    safe_lang,
    translate_to_english,
    translate_from_english
)

import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_path = "tmp_uploaded.pdf"
        with open(tmp_path, "wb") as f:
            f.write(contents)
        text = extract_text_from_pdf(tmp_path)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/verify")
def verify(req: ClaimRequest):
    try:
        # 1) original + detect language
        original_claim = req.claim.strip()
        user_lang = detect_lang(original_claim) or "en"     # from translate.py
        user_lang = safe_lang(user_lang)                    # ensure valid code

        # 2) translate claim -> English (for RAG + NLI)
        claim_en = translate_to_english(original_claim)

        print(f"[verify] user_lang={user_lang} claim_en={claim_en[:150]}")

        # 3) retrieve + rerank (EN)
        retrieved = retrieve_top_facts(claim_en)
        reranked = rerank_with_cross_encoder(claim_en, retrieved)

        print(f"[verify] retrieved {len(reranked)} docs (top idxs: {[d.get('idx') for d in reranked]})")

        # 4) add English summary field (ensure consistent naming)
        for ev in reranked:
            # some docs might already have 'summary' or 'fact_text'
            summary_en = ev.get("summary") or ev.get("fact_text") or ev.get("text") or ""
            ev["summary_en"] = str(summary_en)
            # initialise translated field (fill later)
            ev["summary_translated"] = None

        # 5) sentence-level stance classification (works on English)
        stance_results = classify_stance_ml(claim_en, reranked)  # returns enriched items with best_sentence (EN)

        print(f"[verify] stance_results len={len(stance_results)} example_stance={stance_results[0].get('stance','?') if stance_results else 'n/a'}")

        # 6) Aggregate verdict
        verdict, conf = aggregate_ml_verdict(stance_results)

        # ML fallback trigger
        if verdict == "USE_ML_MODEL":
            reason_text = translate_from_english("Low confidence in RAG evidence. Use ML model.", user_lang)
            return {
                "verdict": "USE_ML_MODEL",
                "confidence": conf,
                "reason": reason_text,
                "evidence": []
            }

        # 7) pick best sentence (english) and translate it
        best_item = max(stance_results, key=lambda x: x.get("stance_confidence", 0))
        best_sentence_en = best_item.get("best_sentence", "") or ""
        best_sentence_translated = translate_from_english(best_sentence_en, user_lang) if best_sentence_en else ""

        # 8) translate each evidence summary back to user's language, robustly
        translated_evidence = []
        for ev in reranked:
            summary_en = ev.get("summary_en", "")
            # try translate; catch failures and mark as unavailable
            try:
                # Always provide translated field (even for English)
                if user_lang == "en":
                    translated = summary_en   # direct passthrough
                else:
                    try:
                        translated = translate_from_english(summary_en, user_lang)
                        if not translated or translated.strip() == summary_en.strip():
                            translated = summary_en
                    except:
                        translated = summary_en
            except Exception as e:
                print(f"[verify] translation failed for idx={ev.get('idx')} error={e}")
                translated = None

            # fallback: if translated is None, keep original English summary so UI can still show something
            translated_evidence.append({
                "idx": ev.get("idx"),
                "score": ev.get("score"),
                "summary_en": summary_en,
                "summary_translated": translated,
                "rerank_score": ev.get("_rerank_score", None),
                # include stance/result if available from stance_results mapping
                "stance": None,
                "stance_confidence": None
            })

        # 9) attach stance info from stance_results (they align by idx in most pipelines)
        # build a quick map by idx to stance result
        stance_map = {}
        for s in stance_results:
            idx = s.get("idx")
            if idx is not None:
                stance_map[int(idx)] = s

        for te in translated_evidence:
            idx = te.get("idx")
            s = stance_map.get(int(idx)) if idx is not None else None
            if s:
                te["stance"] = s.get("stance")
                te["stance_confidence"] = s.get("stance_confidence")
                te["best_sentence_en"] = s.get("best_sentence")
                # Also include best_sentence translated if possible
                try:
                    te["best_sentence_translated"] = translate_from_english(s.get("best_sentence",""), user_lang) if s.get("best_sentence") else None
                except:
                    te["best_sentence_translated"] = None
            else:
                te["stance"] = None
                te["stance_confidence"] = None
                te["best_sentence_en"] = None
                te["best_sentence_translated"] = None

        # 10) Sort evidence by similarity / rerank (optional)
        translated_evidence = sorted(translated_evidence, key=lambda x: (x.get("rerank_score") or 0, -x.get("score", 0)), reverse=True)

        # 11) final response (reason in user language if possible)
        reason_text = best_sentence_translated or translate_from_english(best_sentence_en, user_lang) or ""

        return {
            "verdict": verdict,
            "confidence": conf,
            "reason": reason_text,
            "evidence": translated_evidence
        }

    except Exception as e:
        print("ERROR in /verify:", e)
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "verdict": "ERROR",
            "confidence": 0
        }

