import streamlit as st
import joblib
import os
import numpy as np

MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# ---------- SIMPLE DARK THEME ----------
st.markdown(
    """
    <style>
    .stApp {
        background: #020617;
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .title-center {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #22c55e;
        margin-bottom: 0.1rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .subtitle-center {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 1.8rem;
        font-size: 0.95rem;
    }
    textarea {
        background-color: #020617 !important;
        color: #e5e7eb !important;
        border-radius: 0.75rem !important;
        border: 1px solid #22c55e !important;
        font-family: "Consolas", "Fira Code", monospace !important;
    }
    .stButton>button {
        background: #16a34a;
        color: #e5e7eb;
        border-radius: 999px;
        padding: 0.55rem 2rem;
        border: none;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .stButton>button:hover {
        background: #22c55e;
        color: #020617;
    }
    .result-card-real {
        background-color: #022c22;
        padding: 0.9rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #16a34a;
    }
    .result-card-fake {
        background-color: #3f1519;
        padding: 0.9rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #f97373;
    }
    .explain-card {
        background-color: #030712;
        padding: 0.9rem 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ------------------------------------


def load_model_and_vectorizer():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        st.error("Model or vectorizer not found. Run train_model.py first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def explain_prediction(text, model, vectorizer, predicted_label: str):
    """
    Simple explanation using Logistic Regression coefficients.
    Shows top words pushing towards REAL or FAKE.
    """
    if not hasattr(model, "coef_"):
        return "Explanation not available for this model."

    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
    except Exception:
        return "Explanation not available (could not read feature names)."

    classes = list(model.classes_)
    if "REAL" in classes:
        real_idx = classes.index("REAL")
    else:
        real_idx = 0

    # For binary classification, sklearn usually stores only coef for classes[1]
    if model.coef_.shape[0] == 1:
        coef_real = model.coef_[0]
    else:
        coef_real = model.coef_[real_idx]

    vec = vectorizer.transform([text])
    vec_coo = vec.tocoo()

    contributions = []
    for idx, value in zip(vec_coo.col, vec_coo.data):
        word = feature_names[idx]
        score = value * coef_real[idx]
        contributions.append((word, score))

    if not contributions:
        return "Text is too short for explanation."

    # For REAL prediction -> show words with highest positive impact
    # For FAKE prediction -> show words with most negative impact on REAL
    reverse = True if predicted_label == "REAL" else False
    contributions_sorted = sorted(contributions, key=lambda x: x[1], reverse=reverse)

    # Pick top 6 unique words
    seen = set()
    top_words = []
    for w, s in contributions_sorted:
        if w not in seen:
            seen.add(w)
            top_words.append((w, s))
        if len(top_words) >= 6:
            break

    lines = []
    if predicted_label == "REAL":
        lines.append("**Words pushing this towards REAL:**")
    else:
        lines.append("**Words pushing this towards FAKE (less typical of REAL news):**")

    for word, score in top_words:
        lines.append(f"- `{word}`  (weight: {score:+.3f})")

    return "\n".join(lines)


model, tfidf = load_model_and_vectorizer()

st.markdown('<div class="title-center">FAKE NEWS DETECTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-center">Type a news headline or short article and the system will classify it as <b>REAL</b> or <b>FAKE</b>.</div>',
    unsafe_allow_html=True,
)

text = st.text_area("Enter text here:", height=120)

if st.button("RUN SCAN"):
    if model is None or tfidf is None:
        st.stop()
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        classes = list(model.classes_)
        try:
            idx_fake = classes.index("FAKE")
        except ValueError:
            idx_fake = 0
        try:
            idx_real = classes.index("REAL")
        except ValueError:
            idx_real = 1 if len(classes) > 1 else 0

        conf_fake = proba[idx_fake] * 100
        conf_real = proba[idx_real] * 100

        result_label = str(pred).upper()

        # ----- Result card -----
        if result_label == "REAL":
            st.markdown(
                f'<div class="result-card-real"><b>PREDICTION:</b> REAL</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card-fake"><b>PREDICTION:</b> FAKE</div>',
                unsafe_allow_html=True,
            )

        # ----- Confidence -----
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence REAL", f"{conf_real:.2f}%")
        with col2:
            st.metric("Confidence FAKE", f"{conf_fake:.2f}%")

        # ----- Explanation -----
        explanation = explain_prediction(text, model, tfidf, result_label)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="explain-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Why this prediction?")
        st.markdown(explanation)
        st.markdown("</div>", unsafe_allow_html=True)
