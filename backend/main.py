from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Initialize app
app = FastAPI()

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Request schema
class SymptomRequest(BaseModel):
    symptoms: str

# Simple text cleaning (replaces spaCy)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Root route (test if API works)
@app.get("/")
def home():
    return {"message": "CarePulse AI Backend is running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: SymptomRequest):
    try:
        # Clean input
        cleaned = clean_text(data.symptoms)

        # Transform input
        vector = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vector)[0]

        # Confidence (if available)
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(vector)[0]) * 100
        else:
            confidence = 75.0  # fallback

        # Risk logic (customizable)
        if confidence > 70:
            risk = "HIGH"
        elif confidence > 40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "prediction": str(prediction),
            "confidence": round(confidence, 2),
            "risk": risk
        }

    except Exception as e:
        return {"error": str(e)}