# IE7374 - MLOps Labs

This repository contains lab assignments for the IE7374 MLOps course.

## Labs Completed

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | FastAPI | Completed |
| Lab 2 | GitHub Actions for ML Training & Calibration | Completed |
| Lab 3 | Apache Airflow Pipeline | Completed |
| Lab 4 | Docker Multi-stage Build | Completed |

## Repository Structure

```
MLOps_Labs/
├── .github/
│   └── workflows/
│       ├── lab2_model_training.yml      # Lab 2: Triggers on push
│       └── lab2_model_calibration.yml   # Lab 2: Scheduled calibration
├── fastapi_lab1/                        # Lab 1: FastAPI
│   ├── src/
│   │   ├── main.py
│   │   ├── data.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── model/
│   ├── assets/
│   ├── requirements.txt
│   └── README.md
├── github_lab2/                         # Lab 2: GitHub Actions
│   ├── .github/workflows/               # Copy of workflows (for reference)
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── metrics/
│   └── README.md
├── airflow_lab3/                        # Lab 3: Airflow Pipeline
│   ├── dags/
│   │   ├── airflow_pipeline.py
│   │   ├── src/
│   │   └── model/
│   ├── images/
│   ├── docker-compose.yaml
│   ├── requirements.txt
│   └── README.md
├── docker_lab4/                         # Lab 4: Docker Multi-stage Build
│   ├── src/
│   │   ├── model_training.py
│   │   ├── main.py
│   │   ├── templates/
│   │   └── statics/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── README.md
└── README.md                            # This file
```

## Important Note About GitHub Actions Workflows

GitHub Actions **only recognizes workflows** in `.github/workflows/` at the **repository root**.

For Lab 2, the workflow files are:
- **Active location:** `.github/workflows/` (at repo root) - These actually run
- **Reference copy:** `github_lab2/.github/workflows/` - Kept to show the workflows belong to Lab 2

Both locations contain the same workflow files, but only the ones at the repo root will execute.

---

## Lab 1: FastAPI

**Location:** `fastapi_lab1/`

REST API for Wine classification using FastAPI:
- Health check endpoint (`/`)
- Prediction endpoint (`/predict`) for wine classification
- Pre-trained ML model using sklearn Wine dataset

### How to Run

```bash
cd fastapi_lab1
pip install -r requirements.txt
cd src
python main.py
```

Access API at http://localhost:8000 (Swagger docs at `/docs`)

See `fastapi_lab1/README.md` for detailed instructions.

---

## Lab 2: GitHub Actions for ML Model Training

**Location:** `github_lab2/`

Automated ML pipeline using GitHub Actions for:
- Model training (GradientBoostingClassifier on Breast Cancer dataset)
- Model evaluation (Accuracy, F1, Precision, Recall, ROC-AUC)
- Model calibration (Isotonic regression)
- Model versioning with timestamps

### Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `lab2_model_training.yml` | Push to main (github_lab2 changes) | Trains and evaluates model |
| `lab2_model_calibration.yml` | Daily at midnight UTC / Manual | Calibrates the latest model |

See `github_lab2/README.md` for detailed instructions.

---

## Lab 3: Apache Airflow Pipeline

**Location:** `airflow_lab3/`

ML pipeline orchestrated with Apache Airflow running in Docker:
- Loads Wine dataset (sklearn built-in)
- Preprocesses data with MinMaxScaler
- Trains KNN classifier (finds optimal K via cross-validation)
- Saves and loads model, applies elbow method

### How to Run

```bash
cd airflow_lab3
docker compose up airflow-init
docker compose up -d
```

Access UI at http://localhost:8080 (login: `airflow` / `airflow`)

See `airflow_lab3/README.md` for detailed instructions.

---

## Lab 4: Docker Multi-stage Build

**Location:** `docker_lab4/`

Containerized ML application using Docker multi-stage builds:
- **Stage 1:** Train Gradient Boosting model on Wine dataset
- **Stage 2:** Serve predictions via Flask API
- Docker Compose for service orchestration

### How to Run

**Option 1: Dockerfile**
```bash
cd docker_lab4
docker build -t wine-classifier .
docker run -p 4000:4000 wine-classifier
```

**Option 2: Docker Compose**
```bash
cd docker_lab4
docker compose up
```

Access UI at http://localhost:4000/predict

See `docker_lab4/README.md` for detailed instructions.

---

## Author

Sai Pranav Krovvidi
