from credit_risk.scoring.credit_score import probability_to_score, risk_band


def test_higher_pd_means_lower_score():
    low_risk, high_risk = probability_to_score([0.05, 0.80])
    assert low_risk > high_risk


def test_risk_bands_monotonic():
    assert risk_band(800) == "A"
    assert risk_band(500) == "E"
