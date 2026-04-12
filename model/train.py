import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("chest pain shortness of breath dizziness", 1),
    ("severe chest pressure sweating nausea", 1),
    ("fainting loss of consciousness", 1),
    ("severe abdominal pain fever", 1),

    ("mild headache runny nose sneezing", 0),
    ("sore throat mild cough", 0),
    ("fatigue mild fever body ache", 0),
]

df = pd.DataFrame(data, columns=["text", "label"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

joblib.dump(model, "../backend/model.pkl")
joblib.dump(vectorizer, "../backend/vectorizer.pkl")

print("Model trained successfully")