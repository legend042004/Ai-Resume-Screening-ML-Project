# src/utils.py
import os
import docx
import fitz  # PyMuPDF for PDF
import pandas as pd

from src.preprocess import clean_text

# ------------------
# Extract text from resume files
# ------------------
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = " ".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
    else:
        raise ValueError("Unsupported file type. Please upload .txt, .docx or .pdf")
    return text

# ------------------
# Save results into Excel
# ------------------
def save_results(email: str, phone: str, cleaned_resume: str, prediction: str, excel_path="data/results.xlsx"):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    # Load existing results
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=["Email", "Phone", "Cleaned_Resume", "Predicted_Category"])

    # Append new row
    new_row = {"Email": email, "Phone": phone, "Cleaned_Resume": cleaned_resume, "Predicted_Category": prediction}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save back
    df.to_excel(excel_path, index=False)
    return excel_path
