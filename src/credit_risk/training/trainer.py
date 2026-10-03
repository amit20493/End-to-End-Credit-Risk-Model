from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from credit_risk.config import settings
from credit_risk.data.processing import build_labeled_dataset
from credit_risk.data.schema import TARGET_COLUMN
from credit_risk.features.pipeline import build_preprocess_pipeline
from credit_risk.training.metrics import classification_metrics

logger = logging.getLogger(__name__)


def _candidate_models(quick: bool) -> dict[str, tuple]:
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(
                    max_iter=1000,
                    random_state=settings.random_state,
                    class_weight="balanced",
                ),
                {"clf__C": [1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(
                    random_state=settings.random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
                {"clf__n_estimators": [50], "clf__max_depth": [8]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=settings.random_state,
                class_weight="balanced",
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                random_state=settings.random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [10, 20, None],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }


def _maybe_log_mlflow(name: str, params: dict, metrics: dict, pipeline: Pipeline) -> None:
    if not settings.mlflow_tracking_uri:
        return
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow extras not installed; skipping experiment tracking.")
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")


def train(
    use_synthetic: bool = False,
    quick: bool | None = None,
    output_path: Path | None = None,
) -> dict:
    quick = settings.quick_train if quick is None else quick
    labeled = build_labeled_dataset(use_synthetic=use_synthetic)

    X = labeled.drop(columns=[TARGET_COLUMN])
    y = labeled[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=settings.random_state)
    results: list[dict] = []
    best: dict | None = None

    for name, (estimator, grid) in _candidate_models(quick).items():
        pipeline = Pipeline(
            steps=[
                ("features", build_preprocess_pipeline()),
                ("clf", estimator),
            ]
        )
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1", n_jobs=1 if quick else -1)
        logger.info("Training %s", name)
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        proba = fitted.predict_proba(X_test)[:, 1]
        pred = (proba >= settings.decision_threshold).astype(int)
        metrics = classification_metrics(y_test, pred, proba)
        record = {
            "name": name,
            "params": search.best_params_,
            "cv_f1": float(search.best_score_),
            "metrics": metrics,
            "pipeline": fitted,
        }
        results.append(record)
        _maybe_log_mlflow(name, search.best_params_, metrics, fitted)
        logger.info("%s metrics=%s cv_f1=%.4f", name, metrics, search.best_score_)
        if best is None or record["cv_f1"] > best["cv_f1"]:
            best = record

    if best is None:
        raise RuntimeError("No model was trained.")

    artifact = {
        "pipeline": best["pipeline"],
        "metadata": {
            "model_name": best["name"],
            "trained_at": datetime.now(UTC).isoformat(),
            "params": best["params"],
            "metrics": best["metrics"],
            "cv_f1": best["cv_f1"],
            "decision_threshold": settings.decision_threshold,
            "candidates": [
                {k: v for k, v in item.items() if k != "pipeline"} for item in results
            ],
        },
    }
    dest = output_path or settings.resolve(settings.model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, dest)
    meta_path = dest.with_suffix(".json")
    meta_path.write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")
    logger.info("Saved champion model %s to %s", best["name"], dest)
    return artifact["metadata"]
