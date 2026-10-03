from credit_risk.scoring.predictor import CreditRiskModel
from credit_risk.training.trainer import train


def test_quick_train_writes_artifact(tmp_path, transactions):
    dest = tmp_path / "model.joblib"
    metadata = train(use_synthetic=True, quick=True, output_path=dest)
    assert dest.exists()
    assert "metrics" in metadata
    assert metadata["model_name"] in {"logistic_regression", "random_forest"}

    model = CreditRiskModel.from_path(dest)
    preds = model.predict_frame(transactions.head(8))
    assert len(preds) == 8
    assert 300 <= preds[0]["credit_score"] <= 850
    assert 0 <= preds[0]["probability_of_default"] <= 1
