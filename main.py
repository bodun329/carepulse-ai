from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORS (connect frontend + backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Input):
    X = vectorizer.transform([data.text])
    prob = model.predict_proba(X)[0][1]

    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "risk": risk,
        "score": float(prob)
    }