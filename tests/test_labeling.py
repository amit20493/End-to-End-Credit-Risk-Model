from credit_risk.data.schema import TARGET_COLUMN
from credit_risk.labeling.rfm import assign_high_risk_labels, build_rfm


def test_rfm_one_row_per_customer(transactions):
    rfm = build_rfm(transactions)
    assert rfm["CustomerId"].is_unique
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)


def test_high_risk_label_binary(transactions):
    labeled = assign_high_risk_labels(transactions)
    assert labeled[TARGET_COLUMN].isin([0, 1]).all()
    assert labeled[TARGET_COLUMN].nunique() >= 1
