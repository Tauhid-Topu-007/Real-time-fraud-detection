# 🛡️ Real-Time Fraud Detection System

> An end-to-end machine learning system for detecting potentially fraudulent financial transactions in real time.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6B23)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

## 📌 Overview

This project demonstrates a production-oriented fraud detection workflow covering:

- Transaction data ingestion and validation
- Feature engineering for transactional risk signals
- XGBoost-based fraud classification
- Persisted model and feature-engineering artifacts
- Real-time inference through a FastAPI REST API
- Health, metrics, and model-refresh endpoints
- Interactive application support through Streamlit
- Testing, monitoring, streaming, and deployment-oriented dependencies

The repository is structured so that experimentation, model training, and inference can evolve independently toward a production ML architecture.

## 🏗️ System Architecture

```text
Transaction
    │
    ▼
┌─────────────────────┐
│ Data Ingestion      │
│ & Validation        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │
│ • balance changes   │
│ • risk signals      │
│ • amount statistics │
│ • temporal features │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ XGBoost Classifier  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Fraud Probability   │
│ + Risk Decision     │
└───────┬───────┬─────┘
        │       │
        ▼       ▼
    FastAPI   Streamlit
        │       │
        └───┬───┘
            ▼
       Monitoring /
       Production
```

## 🤖 Model

The repository currently stores an XGBoost model at `models/xgboost_model.pkl` together with the serialized feature engineer and model metadata.

The metadata records **5,489,473 transactions** with a fraud rate of approximately **0.0769%**, highlighting the highly imbalanced nature of the fraud-detection problem. The stored model uses **25 engineered/model features** and records an optimal threshold of **0.26** in the training metadata. fileciteturn20file0

### Feature groups

The stored model metadata includes features such as:

- Transaction step/time
- Transaction amount
- Origin and destination balances
- One-hot transaction types
- Transaction type risk score
- Origin balance change/error
- Amount-to-balance ratio
- Origin transaction count and average amount
- Historical origin fraud ratio
- Destination transaction count and average amount
- Log-transformed amount
- Amount z-score
- Hour / night indicator
- Amount-type risk interaction

These features are designed to capture both transaction-level behavior and account-level patterns. fileciteturn20file0

### Important evaluation note

The metrics stored in `models/model_info.json` are all `1.0` (precision, recall, F1, ROC-AUC and PR-AUC). These should be treated as **training/evaluation metadata rather than a claim of guaranteed real-world performance**. Fraud datasets are highly imbalanced, and leakage, split strategy, threshold selection, and temporal validation can substantially affect results. A production deployment should validate performance on a strictly held-out and preferably time-based test set before relying on these figures. fileciteturn20file0

## 🚀 Project Structure

```text
Real-time-fraud-detection/
│
├── models/
│   ├── feature_engineer.pkl
│   ├── model_info.json
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── inference/
│       └── predictor.py
│
├── scripts/
│   ├── quick_start.py
│   └── train_for_streamlit.py
│
├── docker/
│   └── Dockerfile
│
├── .env.example
├── requirements.txt
├── requirements_streamlit.txt
├── run.py
├── run_training.py
├── setup.py
└── README.md
```

The repository contains dedicated modules for API serving, data ingestion/validation, feature engineering, inference, training utilities, notebooks, Docker setup, and serialized model artifacts. fileciteturn19file0

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tauhid-Topu-007/Real-time-fraud-detection.git
cd Real-time-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The dependency set includes data-science libraries, XGBoost/LightGBM, FastAPI, visualization tools, SHAP, Redis, Kafka, Prometheus/Evidently, testing tools, and supporting utilities. fileciteturn18file0

For the Streamlit application, use:

```bash
pip install -r requirements_streamlit.txt
```

## ▶️ Run the API

The repository provides `run.py` as the API entry point. It loads API settings from `configs/config.yaml` and starts the FastAPI application with Uvicorn. fileciteturn16file0

```bash
python run.py
```

Or run FastAPI directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at:

```text
http://localhost:8000/docs
```

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service status |
| `GET` | `/health` | API/model health check |
| `POST` | `/predict` | Predict fraud probability and risk |
| `GET` | `/metrics` | Retrieve prediction/model metrics |
| `POST` | `/refresh` | Reload model artifacts |

The FastAPI service exposes these endpoints and loads the `FraudPredictor` for inference. fileciteturn21file0

## 📥 Prediction Request

A transaction request accepts fields including:

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 5000.0,
  "nameOrig": "C123456789",
  "oldbalanceOrg": 10000.0,
  "newbalanceOrig": 5000.0,
  "nameDest": "C987654321",
  "oldbalanceDest": 2000.0,
  "newbalanceDest": 7000.0,
  "isFlaggedFraud": 0
}
```

Supported transaction types are `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, and `CASH_IN`. The API schema validates positive transaction amounts, non-negative balances, and valid transaction types. A transaction ID can also be generated automatically when it is not supplied. fileciteturn22file0

Example request with cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 5000,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 5000,
    "nameDest": "C987654321",
    "oldbalanceDest": 2000,
    "newbalanceDest": 7000,
    "isFlaggedFraud": 0
  }'
```

A prediction response contains fields such as:

- `transaction_id`
- `fraud_probability`
- `risk_level`
- `decision`
- `timestamp`
- `model_version`
- `features_used`
- `top_features`

The response schema is defined in `src/api/schemas.py`. fileciteturn22file0

## 📊 Streamlit

The project also includes Streamlit-oriented dependencies and training support for an interactive fraud-detection interface.

```bash
streamlit run streamlit_app/app.py
```

> If the Streamlit application path differs in your current working tree, use the corresponding Streamlit entry point under the project source tree.

## 🧪 Training

Training-related notebooks and scripts are included for reproducible experimentation:

```text
notebooks/01_eda.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_model_training.ipynb
run_training.py
scripts/train_for_streamlit.py
```

A typical ML workflow is:

```text
Raw Transactions
      ↓
EDA & Data Validation
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
XGBoost Training
      ↓
Threshold Selection
      ↓
Evaluation
      ↓
Serialize Model + Feature Engineer
      ↓
FastAPI / Streamlit Inference
```

## 🧰 Production-Oriented Components

The project dependencies and structure leave room for a larger production architecture:

- **FastAPI** — low-latency inference API
- **Redis** — low-latency state/cache layer
- **Kafka** — transaction/event streaming
- **Prometheus** — metrics collection
- **Evidently** — data/model monitoring
- **SHAP** — model explainability
- **Docker** — containerized deployment
- **Pytest** — automated testing

These components are reflected in the repository's dependency configuration. fileciteturn18file0

## 🔐 Security & Production Checklist

Before deploying this system to a real financial environment, consider:

- Replace `allow_origins=["*"]` with an explicit frontend/service allowlist.
- Add authentication and authorization to prediction and model-refresh endpoints.
- Never commit real credentials or sensitive transaction data.
- Keep secrets in environment variables or a secret manager.
- Encrypt sensitive data in transit and at rest.
- Add rate limiting and request-size limits.
- Add structured audit logging and trace IDs.
- Use time-based validation to reduce temporal leakage.
- Monitor precision, recall, PR-AUC, false positives, and false negatives.
- Establish a model rollback/versioning strategy.
- Add human-review workflows for high-risk transactions.
- Recalibrate thresholds as fraud patterns and business costs change.

## ⚠️ Current Limitations / Next Steps

This repository is a strong ML engineering foundation, but several areas should be strengthened before production deployment:

1. Add a reproducible, leakage-safe temporal evaluation pipeline.
2. Benchmark against strong baselines and cost-sensitive metrics.
3. Add automated unit/integration tests for the complete inference path.
4. Add CI/CD with model and API validation.
5. Implement authenticated API access.
6. Complete Docker and deployment configuration.
7. Add real streaming ingestion with Kafka.
8. Add drift monitoring and alerting.
9. Add model registry/version tracking.
10. Document the exact training dataset provenance and preprocessing assumptions.

## 📈 Why Fraud Detection Is Challenging

Fraud detection is a highly imbalanced classification problem. A model can achieve high accuracy while still failing to detect a meaningful number of fraudulent transactions. For this reason, production evaluation should emphasize:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- False-positive rate
- False-negative cost
- Detection latency
- Business loss prevented

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data | Pandas, NumPy, SciPy, Dask, PyArrow |
| ML | XGBoost, LightGBM, Scikit-learn |
| Imbalance | imbalanced-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Streaming | Kafka |
| Cache | Redis |
| Monitoring | Prometheus, Evidently |
| Testing | Pytest, pytest-cov, HTTPX |
| Deployment | Docker |

## 📄 License

This project is released under the MIT License.

## 👨‍💻 Author

**Tauhidul Islam Topu**  
CSE | Machine Learning & AI Enthusiast

GitHub: https://github.com/Tauhid-Topu-007

---

⭐ If this project is useful for learning or research, consider giving the repository a star.
