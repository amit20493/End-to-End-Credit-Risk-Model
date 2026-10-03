from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from credit_risk.config import settings
from credit_risk.data.schema import REQUIRED_COLUMNS
from credit_risk.data.synthetic import generate_synthetic_transactions
from credit_risk.features.pipeline import validate_columns
from credit_risk.labeling.rfm import assign_high_risk_labels

logger = logging.getLogger(__name__)


def load_raw(use_synthetic: bool = False) -> pd.DataFrame:
    raw_path = settings.resolve(settings.raw_data_path)
    if use_synthetic or not raw_path.exists():
        if not use_synthetic:
            logger.warning("Raw data not found at %s; using synthetic transactions.", raw_path)
        return generate_synthetic_transactions(random_state=settings.random_state)
    frame = pd.read_csv(raw_path)
    validate_columns(frame, list(REQUIRED_COLUMNS))
    return frame


def build_labeled_dataset(use_synthetic: bool = False, output_path: Path | None = None) -> pd.DataFrame:
    raw = load_raw(use_synthetic=use_synthetic)
    labeled = assign_high_risk_labels(raw)
    dest = output_path or settings.resolve(settings.target_data_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(dest, index=False)
    logger.info("Wrote labeled dataset n=%s to %s", len(labeled), dest)
    return labeled
