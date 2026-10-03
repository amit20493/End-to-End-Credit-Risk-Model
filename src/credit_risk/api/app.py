from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status

from credit_risk import __version__
from credit_risk.api.deps import verify_api_key
from credit_risk.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from credit_risk.config import settings
from credit_risk.logging_utils import configure_logging
from credit_risk.scoring.predictor import CreditRiskModel, ModelNotLoadedError

configure_logging()
_model: CreditRiskModel | None = None


def get_model() -> CreditRiskModel:
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = CreditRiskModel.from_path()
    except ModelNotLoadedError:
        _model = None
    yield
    _model = None


app = FastAPI(
    title="Credit Risk Scoring API",
    version=__version__,
    description="Predict probability of default, credit score, and risk band.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    meta = _model.metadata if _model else {}
    return HealthResponse(
        status="ok" if _model else "degraded",
        model_loaded=_model is not None,
        model_name=meta.get("model_name"),
        trained_at=meta.get("trained_at"),
    )


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    tags=["scoring"],
    dependencies=[Depends(verify_api_key)],
)
def predict(payload: TransactionRequest) -> PredictionResponse:
    model = get_model()
    frame = pd.DataFrame([payload.model_dump()])
    return PredictionResponse(**model.predict_frame(frame)[0])


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["scoring"],
    dependencies=[Depends(verify_api_key)],
)
def predict_batch(payload: BatchTransactionRequest) -> BatchPredictionResponse:
    model = get_model()
    frame = pd.DataFrame([item.model_dump() for item in payload.transactions])
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**row) for row in model.predict_frame(frame)]
    )


@app.get("/", tags=["ops"])
def root() -> dict:
    return {"service": settings.app_name, "version": __version__}
