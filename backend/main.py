

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from text_utils import clean_text

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Spam Classifier API",
    description="Predicts whether a message is spam or ham (not spam).",
    version="1.0.0",
)

# Allow the Flask frontend (running on a different host/port, e.g. Render)
# to call this API from the browser if needed. Tighten this to your actual
# frontend URL in production if you want to be stricter.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + vectorizer once at startup, not on every request.
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The message to classify")


class PredictResponse(BaseModel):
    label: str
    spam_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    proba = model.predict_proba(vec)[0]  # [P(ham), P(spam)]
    spam_prob = float(proba[1])
    label = "spam" if spam_prob >= 0.5 else "ham"

    return PredictResponse(label=label, spam_probability=round(spam_prob, 4))
