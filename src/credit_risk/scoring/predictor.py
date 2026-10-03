from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import settings
from credit_risk.data.schema import REQUIRED_COLUMNS
from credit_risk.features.pipeline import validate_columns
from credit_risk.scoring.credit_score import probability_to_score, risk_band


class ModelNotLoadedError(RuntimeError):
    pass


class CreditRiskModel:
    def __init__(self, artifact: dict) -> None:
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})
        self.threshold = float(
            self.metadata.get("decision_threshold", settings.decision_threshold)
        )

    @classmethod
    def from_path(cls, path: Path | None = None) -> CreditRiskModel:
        model_path = path or settings.resolve(settings.model_path)
        if not model_path.exists():
            raise ModelNotLoadedError(f"Model artifact not found at {model_path}")
        return cls(joblib.load(model_path))

    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        validate_columns(frame, list(REQUIRED_COLUMNS))
        proba = self.pipeline.predict_proba(frame)[:, 1]
        scores = probability_to_score(proba)
        results = []
        for i, row in enumerate(frame.itertuples(index=False)):
            pd_default = float(proba[i])
            score = int(round(float(scores[i])))
            results.append(
                {
                    "transaction_id": getattr(row, "TransactionId", None),
                    "customer_id": getattr(row, "CustomerId", None),
                    "probability_of_default": round(pd_default, 6),
                    "credit_score": score,
                    "risk_band": risk_band(score),
                    "is_high_risk": bool(pd_default >= self.threshold),
                    "model_name": self.metadata.get("model_name"),
                }
            )
        return results


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> CreditRiskModel:
    path = Path(model_path) if model_path else None
    return CreditRiskModel.from_path(path)
