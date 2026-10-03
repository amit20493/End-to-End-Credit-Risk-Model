import json

import pandas as pd
import pytest

from credit_risk.data.schema import REQUIRED_COLUMNS
from credit_risk.data.synthetic import generate_synthetic_transactions


@pytest.fixture
def transactions() -> pd.DataFrame:
    return generate_synthetic_transactions(n_customers=40, n_transactions=120, random_state=0)


@pytest.fixture
def sample_payload() -> dict:
    frame = generate_synthetic_transactions(n_customers=5, n_transactions=1, random_state=1)
    return json.loads(frame.loc[:, list(REQUIRED_COLUMNS)].to_json(orient="records"))[0]
