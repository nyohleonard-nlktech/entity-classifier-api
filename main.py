from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import MODEL_PATH
from engine import ProductionEntityClassifier

class SingleClassifyRequest(BaseModel):
    text: str

class BatchClassifyRequest(BaseModel):
    texts: List[str]

class ClassificationResult(BaseModel):
    input: str
    label: str
    confidence: float
    source: str

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["classifier"] = ProductionEntityClassifier(model_path=MODEL_PATH)
    yield
    app_state.clear()

app = FastAPI(title="Entity Classifier API", lifespan=lifespan)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    is_ready = app_state.get("classifier") is not None
    return {"status": "healthy" if is_ready else "unhealthy"}

@app.post("/predict", response_model=ClassificationResult, status_code=status.HTTP_200_OK)
async def predict_single(payload: SingleClassifyRequest):
    engine = app_state.get("classifier")
    if not engine:
        raise HTTPException(status_code=503, detail="Engine uninitialized.")
    return engine.predict(payload.text)

@app.post("/predict/batch", response_model=List[ClassificationResult], status_code=status.HTTP_200_OK)
async def predict_batch(payload: BatchClassifyRequest):
    engine = app_state.get("classifier")
    if not engine:
        raise HTTPException(status_code=503, detail="Engine uninitialized.")
    return [engine.predict(text) for text in payload.texts]
