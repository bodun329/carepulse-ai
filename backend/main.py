from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import spacy
from pymongo import MongoClient

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NLP
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if t.is_alpha])

# LOAD MODEL
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# DATABASE (MongoDB local)
client = MongoClient("mongodb://localhost:27017")
db = client["carepulse"]
collection = db["patients"]

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Input):
    cleaned = preprocess(data.text)

    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)[0][1]

    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # SAVE TO DB
    collection.insert_one({
        "text": data.text,
        "cleaned": cleaned,
        "risk": risk,
        "score": float(prob)
    })

    return {
        "risk": risk,
        "score": float(prob)
    }