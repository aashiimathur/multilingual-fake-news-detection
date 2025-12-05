import streamlit as st
import requests
import json
import tempfile

BACKEND_URL = "http://127.0.0.1:8000"

# ================================================================
# PAGE SETUP + PREMIUM DARK THEME
# ================================================================
st.set_page_config(page_title="FactCheck AI", page_icon="📰", layout="wide")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Global */
html, body, div, span {
    font-family: 'Inter', sans-serif !important;
}

body, .main, .block-container {
    background-color: #0d1117 !important;
    color: #e6e7eb !important;
}

header, footer {visibility: hidden !important;}

/* Titles */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 52px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #4EA8DE, #B197FC, #9775FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #fafbfc !important;
}

hr {
    height: 1px;
    border: 0;
    margin: 30px 0;
    background: linear-gradient(to right, #1b1f24, #2d333b, #1b1f24);
}

/* Cards */
.card {
    background: rgba(22, 27, 34, 0.7);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(240, 246, 252, 0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    transition: 0.25s;
}
.card:hover { transform: translateY(-3px); }

/* Badges */
.badge-true {
    background: linear-gradient(135deg, #2ecc71, #1f9f57);
    padding: 14px 26px;
    border-radius: 14px;
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.badge-fake {
    background: linear-gradient(135deg, #ff4c4c, #c62828);
    padding: 14px 26px;
    border-radius: 14px;
    font-size: 26px;
    font-weight: 800;
    color: white;
}

.badge-unverified {
    background: linear-gradient(135deg, #f1c40f, #d68910);
    padding: 14px 26px;
    border-radius: 14px;
    font-size: 26px;
    font-weight: 800;
    color: black;
}

.badge-fallback {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    padding: 14px 26px;
    border-radius: 14px;
    font-size: 26px;
    font-weight: 800;
    color: white;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    border: none;
    color: white;
    padding: 12px 22px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 6px 16px rgba(35, 134, 54, 0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
}

/* Inputs */
textarea, input {
    background-color: #161b22 !important;
    color: #e6e7eb !important;
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #4EA8DE, #9775FA, #B197FC) !important;
}

</style>
""", unsafe_allow_html=True)


# ================================================================
# HEADER
# ================================================================
st.markdown("<h1>📰 FactCheck AI</h1>", unsafe_allow_html=True)
st.markdown("### 🔎 Multilingual RAG • NLI Stance • Cross-Encoders • ML Fallback")
st.write("")

# ================================================================
# PDF UPLOAD
# ================================================================
st.subheader("📄 Upload PDF (Optional)")
pdf = st.file_uploader("Drag & drop a PDF", type=["pdf"])
uploaded_text = ""

if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        path = tmp.name

    with open(path, "rb") as f:
        res = requests.post(f"{BACKEND_URL}/upload-pdf", files={"file": f})

    if res.ok:
        uploaded_text = res.json().get("text", "")
        st.success("PDF extracted successfully!")
        st.text_area("Extracted Text", uploaded_text, height=160)

st.write("---")

# ================================================================
# CLAIM INPUT
# ================================================================
st.subheader("✍️ Enter Claim to Verify")
claim = st.text_area("Enter a claim:", uploaded_text, height=120)

run = st.button("🚀 Run Fact Check")

# ================================================================
# PROCESS CLAIM
# ================================================================
if run:
    if not claim.strip():
        st.error("Please enter a claim first.")
    else:
        with st.spinner("Analyzing with RAG + Stance + ML Fallback..."):
            res = requests.post(f"{BACKEND_URL}/verify", json={"claim": claim})

        if not res.ok:
            st.error("Backend Error!")
            st.text(res.text)
        else:
            data = res.json()

            verdict = data.get("verdict", "")
            conf = data.get("confidence", 0)
            reason = data.get("reason", "")
            evidence = data.get("evidence", [])

            st.write("---")

            # ============================================================
            # VERDICT BADGE
            # ============================================================
            badge_map = {
                "TRUE": "badge-true",
                "FAKE": "badge-fake",
                "USE_ML_MODEL": "badge-fallback",
                "UNVERIFIED": "badge-unverified",
            }

            st.subheader("🏁 Final Verdict")
            cls = badge_map.get(verdict, "badge-unverified")
            st.markdown(f"<span class='{cls}'>{verdict}</span>", unsafe_allow_html=True)

            # ============================================================
            # ML FALLBACK UI (special view)
            # ============================================================
            if verdict == "USE_ML_MODEL":

                st.markdown("### 🤖 ML Fallback Activated")

                st.markdown("""
                    <div class='card' style='border-left: 6px solid #8b5cf6;'>
                        <h4 style='color:#d7c9ff;'>ML Classifier Used Instead of RAG</h4>
                        <p>
                            The RAG retriever found evidence, but it was weak or irrelevant.
                            A high-accuracy NLI classifier was used to determine the factuality.
                        </p>
                        <p style='font-size:13px;color:#9aa0a6;'>
                            Model: <b>MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</b>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # ML confidence bar
                st.markdown("### 🎯 ML Confidence")

                st.markdown(f"""
                    <div class="card" style="border-left:5px solid #8b5cf6;">
                        <div class="progress">
                            <div class="bar" style="width:{conf}%; background:linear-gradient(90deg,#8b5cf6,#6366f1);height:14px;border-radius:8px;"></div>
                        </div>
                        <p style="margin-top:6px;"><b>{conf:.2f}% confidence</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # ML reason
                st.markdown("### 🧠 ML Explanation")
                st.markdown(f"""
                    <div class="card" style="border-left:5px solid #8b5cf6;">
                        {reason}
                    </div>
                """, unsafe_allow_html=True)

                # No evidence section for ML fallback
                st.subheader("🔍 Evidence Skipped")
                st.info("RAG evidence was ignored because it failed evaluation. ML model handled the verdict.")

                st.write("---")

            else:
                # ============================================================
                # NORMAL RAG UI
                # ============================================================
                st.markdown("### 🎯 Confidence Level")
                st.progress(min(conf / 100, 1))
                st.write(f"**{conf:.2f}% confidence**")

                st.markdown("### 🧠 Reason")
                st.markdown(f"<div class='card'>{reason}</div>", unsafe_allow_html=True)

                st.write("---")

                # ============================
                # EVIDENCE SECTION
                # ============================
                st.subheader("🔍 Retrieved Evidence")

                if len(evidence) == 0:
                    st.info("No evidence available.")
                else:
                    cols_per_row = 3
                    for i in range(0, len(evidence), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col, ev in zip(cols, evidence[i:i+cols_per_row]):
                            text = (
                                ev.get("summary_translated")
                                or ev.get("summary_en")
                                or "(translation unavailable)"
                            )
                            col.markdown(
                                f"""
                                <div class="card">
                                    <h4>📌 Evidence</h4>
                                    <p>{text}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.write("---")

            # ============================================================
            # RAW JSON VIEW
            # ============================================================
            with st.expander("🛠 Developer View (Raw JSON Response)"):
                st.json(data)
