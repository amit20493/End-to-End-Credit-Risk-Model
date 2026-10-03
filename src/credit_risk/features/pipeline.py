from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data.schema import CATEGORICAL_COLUMNS, ENGINEERED_NUMERIC, NUMERIC_COLUMNS
from credit_risk.features.transformers import FeatureEngineer


def build_preprocess_pipeline() -> Pipeline:
    numeric_cols = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocess", preprocessor),
        ]
    )


def validate_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
