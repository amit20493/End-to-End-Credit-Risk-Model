from __future__ import annotations

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from credit_risk.config import settings
from credit_risk.data.schema import CUSTOMER_COLUMN, TARGET_COLUMN, TIME_COLUMN

logger = logging.getLogger(__name__)


def build_rfm(raw: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    frame = raw.copy()
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True)
    if snapshot is None:
        snapshot = frame[TIME_COLUMN].max() + pd.Timedelta(days=1)

    rfm = (
        frame.groupby(CUSTOMER_COLUMN)
        .agg(
            Recency=(TIME_COLUMN, lambda x: (snapshot - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    rfm["Frequency"] = rfm["Frequency"].clip(lower=1)
    cap = rfm["Monetary"].abs().quantile(0.99)
    rfm["Monetary"] = rfm["Monetary"].clip(lower=-cap, upper=cap)
    return rfm


def assign_high_risk_labels(raw: pd.DataFrame, n_clusters: int | None = None) -> pd.DataFrame:
    """Proxy default labels via RFM clustering (Xente has no observed default)."""
    n_clusters = n_clusters or settings.n_risk_clusters
    rfm = build_rfm(raw)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=settings.random_state,
        n_init=10,
    )
    rfm["Cluster"] = kmeans.fit_predict(scaled)

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=["Recency", "Frequency", "Monetary"],
    )
    # High risk: stale, infrequent, low/negative monetary engagement.
    risk_score = centers["Recency"] - centers["Frequency"] - centers["Monetary"].abs()
    high_risk_cluster = int(risk_score.idxmax())
    rfm[TARGET_COLUMN] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    logger.info("High-risk cluster=%s sizes=%s", high_risk_cluster, rfm["Cluster"].value_counts().to_dict())
    labeled = raw.merge(rfm[[CUSTOMER_COLUMN, TARGET_COLUMN]], on=CUSTOMER_COLUMN, how="left")
    labeled[TARGET_COLUMN] = labeled[TARGET_COLUMN].fillna(0).astype(int)
    return labeled
