from pathlib import Path

import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Toxic Comment Detector")

BASE = Path(__file__).parent
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
EXAMPLES = [
    "Thank you for fixing the article, great work!",
    "You are a complete idiot and nobody wants your edits.",
    "This is stupid garbage, stop wasting everyone's time.",
]


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "toxic_model.pkl", "rb"))


@st.cache_resource
def load_vectorizer():
    return pickle.load(open(BASE / "toxic_vectorizer.pkl", "rb"))


model = load_model()
vectorizer = load_vectorizer()

st.title("Toxic Comment Detector")
st.write(
    "A TF-IDF + Logistic Regression model (trained on the Kaggle Jigsaw Toxic Comment Classification dataset) "
    "estimates the probability that a comment is toxic, severely toxic, obscene, a threat, an insult, or hateful "
    "towards an identity group. A comment can have several labels at once."
)
st.info("The training data comes from real online comments and contains offensive language. This is a modeling demo, not a moderation tool.")

choice = st.selectbox("Try an example (optional)", ["Write my own"] + EXAMPLES)
text = st.text_area("Comment", "" if choice == "Write my own" else choice, height=100)

if st.button("Analyze") and text.strip():
    proba = model.predict_proba(vectorizer.transform([text]))[0]
    scores = pd.Series(proba, index=LABELS, name="probability")
    flagged = [label for label, p in scores.items() if p >= 0.5]
    if flagged:
        st.error("Flagged as: **" + ", ".join(flagged) + "**")
    else:
        st.success("No label reached 50% probability.")
    st.bar_chart(scores)
    st.dataframe(scores.map(lambda p: f"{p:.1%}").to_frame("probability"), width="stretch")

st.caption(
    "Model: one logistic regression per label on TF-IDF features (validation mean AUC ≈ 0.979). Rare labels "
    "such as 'threat' and 'identity_hate' are the hardest for this kind of model, and it can mistake quotes or "
    "discussions about offensive words for the words themselves."
)
