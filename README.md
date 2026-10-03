# End-to-End Credit Risk Model

Production credit-risk scoring service for Xente-style mobile-money transactions. The pipeline builds a **proxy default label** with RFM + K-Means (the public dataset has no observed default), trains a sklearn model, maps **probability of default (PD)** to a **credit score** and **risk band**, and serves predictions through FastAPI.

## Architecture

1. **Feature engineering** — calendar features from `TransactionStartTime` and customer-level amount aggregates fitted on train data only (new customers get training medians).
2. **Target** — RFM recency / frequency / monetary clustering; the lowest-engagement cluster is labeled high risk.
3. **Models** — logistic regression and random forest with stratified CV; the champion (best CV F1) is serialized with the full preprocess pipeline.
4. **Scoring** — PD → scorecard (`PDO` / base odds) → bands A–E.
5. **Serving** — FastAPI `/v1/predict` and `/v1/predict/batch`.
6. **CI/CD** — GitHub Actions unit tests on PRs; UAT pipeline on `main` (train, container smoke tests, push `uat` image to GHCR). No production deploy.

## Project layout

```text
src/credit_risk/     # installable package (features, labeling, training, API)
scripts/             # process / train / serve entrypoints
tests/               # unit and API tests (synthetic Xente-shaped data)
.github/workflows/   # CI and CD
artifacts/           # model.joblib (created by training)
data/raw/            # place Xente data.csv here
```

## Local setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put the Xente file at `data/raw/data.csv`. If it is missing, training falls back to synthetic data (used in CI).

```bash
python -m credit_risk.cli process
python -m credit_risk.cli train --quick
python -m credit_risk.cli serve
```

Full grid search (production training):

```bash
python -m credit_risk.cli train
```

Optional MLflow: set `MLFLOW_TRACKING_URI` and install `pip install -e ".[mlflow]"`.

API docs: http://127.0.0.1:8000/docs  

Health: `GET /health`  ·  Ready: `GET /ready`

### Example score request

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"TransactionId\": \"TransactionId_1\",
    \"BatchId\": \"BatchId_1\",
    \"AccountId\": \"AccountId_1\",
    \"SubscriptionId\": \"SubscriptionId_1\",
    \"CustomerId\": \"CustomerId_1\",
    \"CurrencyCode\": \"UGX\",
    \"CountryCode\": 256,
    \"ProviderId\": \"ProviderId_1\",
    \"ProductId\": \"ProductId_1\",
    \"ProductCategory\": \"airtime\",
    \"ChannelId\": \"ChannelId_1\",
    \"Amount\": 1000,
    \"Value\": 1000,
    \"TransactionStartTime\": \"2018-11-15T12:00:00Z\",
    \"PricingStrategy\": 2,
    \"FraudResult\": 0
  }"
```

Response fields: `probability_of_default`, `credit_score` (300–850), `risk_band` (A–E), `is_high_risk`.

If `API_KEY` is set in the environment, send it as header `X-API-Key`.

## Tests

```bash
ruff check src tests scripts
pytest
```

## Docker

Train a model first so `artifacts/model.joblib` exists, then:

```bash
docker compose up --build
```

Or:

```bash
docker build -t credit-risk-api:latest .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/artifacts/model.joblib credit-risk-api:latest
```

## GitHub CI/CD (stops at UAT)

Pipeline: **PR quality → merge to `main` → UAT image + smoke tests → GHCR `uat` tag**. There is no production SSH deploy and no `DEPLOY_*` secrets.

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PRs and pushes to `main` | Ruff + pytest + coverage |
| `.github/workflows/cd.yml` (`UAT`) | Push to `main` or **Run workflow** | Re-run tests, train UAT model, Docker smoke (`/health`, `/ready`, `/v1/predict`), push `ghcr.io/<owner>/<repo>:uat` |

Enable GitHub Container Registry: repo **Settings → Actions → General → Workflow permissions → Read and write**.

Create GitHub environment **uat**: **Settings → Environments → New environment** → name `uat`.

UAT image (this repo): `ghcr.io/amit20493/end-to-end-credit-risk-model:uat`

Packages are private by default. To pull locally: `echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin` (`read:packages` token).

## How to test (local + UAT)

### 1. Unit / API tests (same as CI)

```bash
pip install -e ".[dev]"
ruff check src tests scripts
pytest
```

### 2. Train then serve locally

```bash
python -m credit_risk.cli train --synthetic --quick
python -m credit_risk.cli serve
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
python scripts/uat_smoke.py
```

If `.env` has `API_KEY` set, smoke uses it automatically. Override:

```bash
# PowerShell
$env:API_KEY="uat-local-key"
python scripts/uat_smoke.py
```

Browser: http://127.0.0.1:8000/docs

### 3. UAT Docker (no VM)

Train first so `artifacts/model.joblib` exists, then:

```bash
docker compose -f docker-compose.uat.yml up --build
```

Second terminal:

```bash
$env:API_KEY="uat-local-key"
python scripts/uat_smoke.py
```

### 4. After GitHub UAT workflow is green

1. Open **Actions → UAT** and confirm the `uat` job passed.
2. Open **Packages** and confirm tag `uat`.
3. Pull and smoke locally:

```bash
docker login ghcr.io
docker pull ghcr.io/amit20493/end-to-end-credit-risk-model:uat
docker run --rm -p 8000:8000 -e APP_ENV=uat -e API_KEY=uat-local-key ghcr.io/amit20493/end-to-end-credit-risk-model:uat
```

Then run `python scripts/uat_smoke.py` with the same `API_KEY`.

UAT smoke checks:

- `GET /health` → `model_loaded: true`
- `GET /ready` → 200
- `POST /v1/predict` without key → 401 when `API_KEY` is set
- `POST /v1/predict` with `X-API-Key` → PD, credit score, band A–E

## Steps to promote a model into UAT

### 1. Prepare data and train

1. Copy `data/raw/data.csv` (Xente transactions).
2. Train on a machine with enough RAM (full RF grid is heavier than `--quick`):

```bash
pip install -e .
python -m credit_risk.cli train
```

3. Confirm `artifacts/model.joblib` and `artifacts/model.json` (metrics, threshold, champion name).

### 2. Review credit-risk quality gates

Check `artifacts/model.json` before promotion:

- **ROC-AUC** and **F1** on the held-out test split
- Positive-class **recall** (missed high-risk customers)
- Champion model name and hyperparameters

Treat RFM labels as a **proxy**, not observed default. Recalibrate PD and the decision threshold when you have real default outcomes.

### 3. Push to `main` for UAT

CI + UAT workflows run automatically. Accept the build only if Actions **UAT** is green and Packages shows `:uat`.

GitHub UAT currently trains with `--synthetic --quick` so the automated image is for pipeline acceptance, not a live credit book. For a realistic UAT model, train locally on `data/raw/data.csv` and run `docker compose -f docker-compose.uat.yml up --build`.

### 4. Model updates in UAT

Retrain → merge to `main` → wait for the `uat` tag → pull that image and re-run `scripts/uat_smoke.py`. Workers load the artifact at process start; restart the container after a new model.

## Risk bands

| Band | Score | Meaning |
|---|---|---|
| A | ≥ 750 | Low PD |
| B | ≥ 680 | Moderate-low |
| C | ≥ 620 | Moderate |
| D | ≥ 560 | Elevated |
| E | < 560 | High risk |

Threshold for `is_high_risk` defaults to PD ≥ `0.5` (`DECISION_THRESHOLD`). Tune on a cost-sensitive cutoff (false approve vs false decline) once you have loss data.
