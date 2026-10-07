"""
FastAPI ML service

  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.config import DEFAULT_CHECKPOINT, load_settings
from api.inference import Predictor

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    settings = load_settings()
    predictor = Predictor(checkpoint_path=settings.checkpoint_path)
    yield


app = FastAPI(title="AI Image Detection API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "loaded": predictor is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload must be an image file")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 15MB)")
    try:
        return predictor.predict_bytes(data)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e
