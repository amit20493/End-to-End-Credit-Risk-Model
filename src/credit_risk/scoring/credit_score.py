from __future__ import annotations

import numpy as np

from credit_risk.config import settings


def probability_to_score(pd: np.ndarray | float) -> np.ndarray:
    """Map probability of default to a bounded credit score (higher = safer)."""
    p = np.clip(np.asarray(pd, dtype=float), 1e-6, 1 - 1e-6)
    factor = settings.score_pdo / np.log(2)
    offset = settings.score_base - factor * np.log(settings.score_base_odds)
    odds = (1.0 - p) / p
    score = offset + factor * np.log(odds)
    return np.clip(score, settings.score_min, settings.score_max)


def risk_band(score: float) -> str:
    if score >= 750:
        return "A"
    if score >= 680:
        return "B"
    if score >= 620:
        return "C"
    if score >= 560:
        return "D"
    return "E"
