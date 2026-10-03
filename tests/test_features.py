from credit_risk.data.schema import ENGINEERED_NUMERIC
from credit_risk.features.transformers import FeatureEngineer


def test_feature_engineer_adds_columns(transactions):
    engineer = FeatureEngineer().fit(transactions)
    transformed = engineer.transform(transactions)
    for col in ENGINEERED_NUMERIC:
        assert col in transformed.columns
    assert transformed["TransactionHour"].between(0, 23).all()
    assert transformed["StdTransactionAmount"].notna().all()


def test_unknown_customer_uses_defaults(transactions):
    engineer = FeatureEngineer().fit(transactions)
    unseen = transactions.iloc[[0]].copy()
    unseen["CustomerId"] = "CustomerId_unknown"
    out = engineer.transform(unseen)
    assert out["TransactionCount"].notna().all()
