import pandas as pd
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if t.is_alpha])

# LOAD DATASET (replace with Kaggle CSV later)
df = pd.DataFrame([
    ("chest pain shortness of breath dizziness", 1),
    ("fainting loss of consciousness", 1),
    ("severe abdominal pain fever", 1),
    ("mild headache runny nose", 0),
    ("sore throat cough fatigue", 0),
], columns=["text", "label"])

df["text"] = df["text"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

base_model = LogisticRegression()
model = CalibratedClassifierCV(base_model, method="sigmoid")
model.fit(X, y)

joblib.dump(model, "../backend/model.pkl")
joblib.dump(vectorizer, "../backend/vectorizer.pkl")

print("✅ Model trained + calibrated successfully")