from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.data.schema import (
    AMOUNT_COLUMN,
    CUSTOMER_COLUMN,
    TIME_COLUMN,
)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    raise TypeError("Feature transformers require a pandas DataFrame.")


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Parse transaction timestamps into calendar features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _as_frame(X)
        ts = pd.to_datetime(frame[TIME_COLUMN], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError("TransactionStartTime contains unparseable values.")
        frame["TransactionHour"] = ts.dt.hour.astype(int)
        frame["TransactionDay"] = ts.dt.day.astype(int)
        frame["TransactionMonth"] = ts.dt.month.astype(int)
        frame["TransactionYear"] = ts.dt.year.astype(int)
        return frame


class CustomerAggregateExtractor(BaseEstimator, TransformerMixin):
    """Customer-level amount stats learned at fit time (no train/serve leakage)."""

    def __init__(self) -> None:
        self.customer_stats_: pd.DataFrame | None = None
        self.global_defaults_: dict[str, float] | None = None

    def fit(self, X: pd.DataFrame, y=None):
        grouped = X.groupby(CUSTOMER_COLUMN)[AMOUNT_COLUMN].agg(["sum", "mean", "count", "std"])
        grouped = grouped.rename(
            columns={
                "sum": "TotalTransactionAmount",
                "mean": "AverageTransactionAmount",
                "count": "TransactionCount",
                "std": "StdTransactionAmount",
            }
        )
        grouped["StdTransactionAmount"] = grouped["StdTransactionAmount"].fillna(0.0)
        self.customer_stats_ = grouped.reset_index()
        self.global_defaults_ = {
            "TotalTransactionAmount": float(grouped["TotalTransactionAmount"].median()),
            "AverageTransactionAmount": float(grouped["AverageTransactionAmount"].median()),
            "TransactionCount": float(grouped["TransactionCount"].median()),
            "StdTransactionAmount": 0.0,
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.customer_stats_ is None or self.global_defaults_ is None:
            raise RuntimeError("CustomerAggregateExtractor must be fitted before transform.")
        frame = X.merge(self.customer_stats_, on=CUSTOMER_COLUMN, how="left")
        for col, default in self.global_defaults_.items():
            frame[col] = frame[col].fillna(default)
        return frame


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Time features plus customer aggregates in one pandas-preserving step."""

    def __init__(self) -> None:
        self.time_ = TimeFeatureExtractor()
        self.agg_ = CustomerAggregateExtractor()

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.time_.fit(frame, y)
        timed = self.time_.transform(frame)
        self.agg_.fit(timed, y)
        return self

    def transform(self, X) -> pd.DataFrame:
        frame = _as_frame(X)
        return self.agg_.transform(self.time_.transform(frame))
