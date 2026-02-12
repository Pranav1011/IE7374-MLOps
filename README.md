# IE7374 - MLOps Labs

This repository contains lab assignments for the IE7374 MLOps course.

## Labs Completed

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 1 | FastAPI | Completed |
| Lab 2 | GitHub Actions for ML Training & Calibration | Completed |

## Repository Structure

```
MLOps_Labs/
├── .github/
│   └── workflows/
│       ├── lab2_model_training.yml      # Lab 2: Triggers on push
│       └── lab2_model_calibration.yml   # Lab 2: Scheduled calibration
├── fastapi_lab1/                        # Lab 1: FastAPI
├── github_lab2/                         # Lab 2: GitHub Actions
│   ├── .github/workflows/               # Copy of workflows (for reference)
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── metrics/
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

Basic FastAPI application demonstrating REST API development.

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

## Author

Sai Pranav Krovvidi
