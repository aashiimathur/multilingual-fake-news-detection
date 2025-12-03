import streamlit as st
import requests
import json
import tempfile
import os

BACKEND_URL = "http://127.0.0.1:8000"

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(
    page_title="Multilingual Fact Checker",
    layout="centered",
)

# Title
st.markdown("""
# 📰 FactCheck AI  
### Multilingual RAG + Cross-Encoder Reranking + NLI Stance Engine
""")

st.write("Upload a PDF or enter a claim below. The system supports **all Indian languages**.")

# ------------------------------
# PDF Upload
# ------------------------------
st.subheader("📄 Upload PDF (optional)")

pdf_file = st.file_uploader("Drop a PDF here", type=["pdf"])
uploaded_text = ""

if pdf_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        res = requests.post(f"{BACKEND_URL}/upload-pdf", files={"file": f})

    if res.status_code == 200:
        data = res.json()
        uploaded_text = data.get("text", "")
        st.success("PDF extracted successfully!")
        st.text_area("Extracted Text", uploaded_text, height=200)


# ------------------------------
# Claim Input
# ------------------------------
st.subheader("✍️ Enter Claim")
claim = st.text_area("Paste your claim:", uploaded_text, height=120)

run_button = st.button("Run Verification")


# ------------------------------
# Run Backend Verification
# ------------------------------
if run_button:
    if not claim.strip():
        st.error("Please enter a claim first.")
    else:
        with st.spinner("Analyzing..."):
            payload = {"claim": claim}
            res = requests.post(f"{BACKEND_URL}/verify", json=payload)

        if res.status_code != 200:
            st.error("Backend returned an error")
            st.text(res.text)
        else:
            data = res.json()

            verdict = data.get("verdict", "UNVERIFIED")
            conf = data.get("confidence", 0)
            reason = data.get("reason", "")
            evidence = data.get("evidence", [])

            # ------------------------------
            # Verdict Display
            # ------------------------------
            st.markdown("---")

            if verdict == "TRUE":
                st.markdown(f"## 🟩 TRUE — **{conf:.2f}%**")
            elif verdict == "FAKE":
                st.markdown(f"## 🟥 FAKE — **{conf:.2f}%**")
            else:
                st.markdown(f"## 🟧 UNVERIFIED — **{conf:.2f}%**")

            st.markdown("### 📝 Reasoning")
            st.write(reason)

            st.markdown("---")

            # ------------------------------
            # Evidence Display (3 per row)
            # ------------------------------
            st.markdown("## 🔍 Evidence (Translated)")

            if len(evidence) == 0:
                st.info("No evidence found.")
            else:
                # Create rows of 3 columns
                for row_start in range(0, len(evidence), 3):
                    row_evidence = evidence[row_start:row_start + 3]
                    cols = st.columns(len(row_evidence))

                    for col, ev in zip(cols, row_evidence):
                        with col:
                            st.markdown("##### 📌 Evidence")

                            # Evidence text fallback ordering
                            translated_summary = (
                                ev.get("summary_translated")
                                or ev.get("summary_origlang")
                                or ev.get("summary_en")
                                or "(translation unavailable)"
                            )

                            # Card-like container
                            with st.container():
                                st.write(translated_summary)

                            st.markdown("---")
