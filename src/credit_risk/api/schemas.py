from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: datetime
    PricingStrategy: int
    FraudResult: int = Field(ge=0, le=1)

    @field_validator("Value")
    @classmethod
    def value_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    transaction_id: str | None
    customer_id: str | None
    probability_of_default: float
    credit_score: int
    risk_band: str
    is_high_risk: bool
    model_name: str | None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    trained_at: str | None = None
