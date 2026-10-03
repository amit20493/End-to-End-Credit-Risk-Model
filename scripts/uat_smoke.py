"""HTTP smoke checks for the UAT scoring API (no production deploy)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

SAMPLE = {
    "TransactionId": "TransactionId_1",
    "BatchId": "BatchId_1",
    "AccountId": "AccountId_1",
    "SubscriptionId": "SubscriptionId_1",
    "CustomerId": "CustomerId_1",
    "CurrencyCode": "UGX",
    "CountryCode": 256,
    "ProviderId": "ProviderId_1",
    "ProductId": "ProductId_1",
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_1",
    "Amount": 1000,
    "Value": 1000,
    "TransactionStartTime": "2018-11-15T12:00:00Z",
    "PricingStrategy": 2,
    "FraudResult": 0,
}


def _get(url: str, headers: dict | None = None) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _post(url: str, payload: dict, headers: dict | None = None) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode()
    merged = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=merged, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def main() -> int:
    base = os.environ.get("UAT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    auth = {"X-API-Key": api_key} if api_key else {}

    code, health = _get(f"{base}/health")
    if code != 200 or not isinstance(health, dict) or health.get("model_loaded") is not True:
        print(f"FAIL /health: {code} {health}")
        return 1
    print(f"OK /health model={health.get('model_name')}")

    code, ready = _get(f"{base}/ready")
    if code != 200:
        print(f"FAIL /ready: {code} {ready}")
        return 1
    print("OK /ready")

    if api_key:
        code, denied = _post(f"{base}/v1/predict", SAMPLE)
        if code != 401:
            print(f"FAIL unauthenticated predict expected 401, got {code} {denied}")
            return 1
        print("OK /v1/predict rejects missing API key")

    code, scored = _post(f"{base}/v1/predict", SAMPLE, headers=auth)
    if code != 200 or not isinstance(scored, dict):
        print(f"FAIL /v1/predict: {code} {scored}")
        return 1
    required = {"probability_of_default", "credit_score", "risk_band", "is_high_risk"}
    missing = required - scored.keys()
    if missing:
        print(f"FAIL /v1/predict missing fields {missing}: {scored}")
        return 1
    if scored["risk_band"] not in {"A", "B", "C", "D", "E"}:
        print(f"FAIL unexpected risk_band {scored['risk_band']}")
        return 1
    print(
        "OK /v1/predict "
        f"pd={scored['probability_of_default']:.4f} "
        f"score={scored['credit_score']} band={scored['risk_band']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
