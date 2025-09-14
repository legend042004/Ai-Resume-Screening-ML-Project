# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    _ = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet')

STOP = set(stopwords.words('english'))
LEM  = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # keep only alphabets
    text = text.lower()
    words = text.split()
    words = [LEM.lemmatize(w) for w in words if w not in STOP and len(w) > 2]
    return " ".join(words)
