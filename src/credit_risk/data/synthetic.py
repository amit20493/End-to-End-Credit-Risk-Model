from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.data.schema import REQUIRED_COLUMNS


def generate_synthetic_transactions(
    n_customers: int = 80,
    n_transactions: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create Xente-shaped data for tests and CI training."""
    rng = np.random.default_rng(random_state)
    customer_ids = [f"CustomerId_{i}" for i in range(n_customers)]
    rows: list[dict] = []

    for i in range(n_transactions):
        cid = customer_ids[int(rng.integers(0, n_customers))]
        amount = float(rng.normal(500, 250))
        rows.append(
            {
                "TransactionId": f"TransactionId_{i}",
                "BatchId": f"BatchId_{i // 10}",
                "AccountId": f"AccountId_{cid.split('_')[1]}",
                "SubscriptionId": f"SubscriptionId_{cid.split('_')[1]}",
                "CustomerId": cid,
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProviderId": f"ProviderId_{int(rng.integers(1, 7))}",
                "ProductId": f"ProductId_{int(rng.integers(1, 16))}",
                "ProductCategory": rng.choice(
                    ["airtime", "data_bundles", "tv", "utility_bill", "financial_services"]
                ),
                "ChannelId": f"ChannelId_{int(rng.integers(1, 6))}",
                "Amount": amount,
                "Value": abs(amount),
                "TransactionStartTime": pd.Timestamp("2018-11-01")
                + pd.Timedelta(days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24))),
                "PricingStrategy": int(rng.integers(0, 5)),
                "FraudResult": int(rng.random() < 0.02),
            }
        )

    df = pd.DataFrame(rows)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Synthetic data missing columns: {missing}")
    return df
