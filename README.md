# End-to-End Credit Risk Model

Production credit-risk scoring service for Xente-style mobile-money transactions. The pipeline builds a **proxy default label** with RFM + K-Means (the public dataset has no observed default), trains a sklearn model, maps **probability of default (PD)** to a **credit score** and **risk band**, and serves predictions through FastAPI.

## Architecture

1. **Feature engineering** — calendar features from `TransactionStartTime` and customer-level amount aggregates fitted on train data only (new customers get training medians).
2. **Target** — RFM recency / frequency / monetary clustering; the lowest-engagement cluster is labeled high risk.
3. **Models** — logistic regression and random forest with stratified CV; the champion (best CV F1) is serialized with the full preprocess pipeline.
4. **Scoring** — PD → scorecard (`PDO` / base odds) → bands A–E.
5. **Serving** — FastAPI `/v1/predict` and `/v1/predict/batch`.
6. **CI/CD** — GitHub Actions lint/test on PRs; train + Docker image push to GHCR on `main`; optional SSH deploy on version tags.

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

## GitHub CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | PRs and pushes to `main` | Install, Ruff, pytest + coverage |
| `.github/workflows/cd.yml` | Push to `main` and tags `v*` | Train artifact, build image, push to `ghcr.io/<owner>/<repo>` |
| CD `deploy` job | Tags `v*` only | SSH pull + `docker compose up` when `DEPLOY_HOST` is set |

Enable GitHub Container Registry: repo **Settings → Actions → General → Workflow permissions → Read and write**.

Packages are private by default. On the server: `echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin`.

### Optional production SSH deploy

Repository **Settings → Secrets and variables**:

- Secret `DEPLOY_USER`
- Secret `DEPLOY_SSH_KEY` (private key)
- Variable `DEPLOY_HOST` (server IP or DNS)
- Variable `DEPLOY_PATH` (directory with `docker-compose.yml`, default `/opt/credit-risk`)

Create a GitHub release tag (`v1.0.0`) to ship.

## Steps to deploy the model

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

### 3. Ship the API image

**A. Local / VM**

```bash
docker compose up -d --build
curl http://127.0.0.1:8000/ready
```

Point a reverse proxy (Nginx / Caddy) at port 8000 with TLS.

**B. GitHub Container Registry**

1. Push `main` (CD builds and pushes the image).
2. On the host:

```bash
docker login ghcr.io
docker pull ghcr.io/<owner>/End-to-End-Credit-Risk-Model:latest
docker tag ghcr.io/<owner>/End-to-End-Credit-Risk-Model:latest credit-risk-api:latest
docker compose up -d --no-build
```

**C. Cloud container services** (same image)

- **AWS**: push to ECR or pull GHCR; run on ECS Fargate / App Runner; attach an ALB; store `model.joblib` in S3 or bake it into the image.
- **Azure**: Azure Container Apps or App Service (container); set `MODEL_PATH` and `API_KEY`.
- **GCP**: Cloud Run; memory ≥ 1Gi; set concurrency and min instances for latency.
- **Render / Railway / Fly.io**: deploy from the Dockerfile; add disk or bake the artifact.

Set environment variables: `APP_ENV=production`, `MODEL_PATH`, `API_KEY`, `DECISION_THRESHOLD`.

### 4. Runtime checks

- `GET /health` — process up; `model_loaded` must be true
- `GET /ready` — 200 before attaching load balancer
- Score a known fixture transaction and compare PD / score to the training notebook
- Enforce `API_KEY` on public endpoints

### 5. Model updates

Retrain → bump image tag (`v1.1.0`) → deploy with rolling restart → keep the previous image for rollback. Do not hot-swap `model.joblib` without restarting workers; they load the artifact at process start.

## Risk bands

| Band | Score | Meaning |
|---|---|---|
| A | ≥ 750 | Low PD |
| B | ≥ 680 | Moderate-low |
| C | ≥ 620 | Moderate |
| D | ≥ 560 | Elevated |
| E | < 560 | High risk |

Threshold for `is_high_risk` defaults to PD ≥ `0.5` (`DECISION_THRESHOLD`). Tune on a cost-sensitive cutoff (false approve vs false decline) once you have loss data.
