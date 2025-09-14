# app.py
import streamlit as st
import joblib
import os
from src.preprocess import clean_text
from src.utils import extract_text, save_results

import sys

# Force Python to see the project root (resumeapp/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ------------------
# Load Model and Vectorizer
# ------------------
with open("models/log_reg_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(page_title="Resume Classifier", page_icon="📄")

st.title("📄 Resume Classification App")
st.write("Upload a resume or paste text below. Enter email & phone, and the app will classify expertise.")

# Email & Phone
email = st.text_input("Enter Candidate Email")
phone = st.text_input("Enter Candidate Phone")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
resume_text = ""

if uploaded_file is not None:
    uploads_dir = "data/uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    resume_text = extract_text(file_path)

# Text Area (optional direct paste)
manual_text = st.text_area("Or Paste Resume Text", height=200)

if manual_text.strip():
    resume_text = manual_text

# Classify Button
if st.button("Classify Resume"):
    if not email or not phone:
        st.warning("⚠️ Please enter both Email and Phone.")
    elif not resume_text.strip():
        st.warning("⚠️ Please upload or paste resume text.")
    else:
        # Preprocess
        cleaned_text = clean_text(resume_text)

        # Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        proba = model.predict_proba(vectorized_text).max() * 100

        # Display
        st.success(f"✅ Predicted Category: **{prediction}** ({proba:.2f}% confidence)")

        # Save to Excel
        excel_path = save_results(email, phone, cleaned_text, prediction)
        st.info(f"📊 Results saved to {excel_path}")
