# Lab 2: GitHub Actions for Model Training, Evaluation & Calibration

This lab demonstrates how to use GitHub Actions to automate machine learning workflows including model training, evaluation, versioning, and probability calibration.

## Implementation Details

| Component | Choice |
|-----------|--------|
| **Dataset** | Breast Cancer Wisconsin (binary classification, 569 samples, 30 features) |
| **Model** | GradientBoostingClassifier (100 estimators, learning_rate=0.1, max_depth=3) |
| **Tracking** | MLflow for experiment tracking |
| **Metrics** | Accuracy, F1 (weighted), Precision, Recall, ROC-AUC, Confusion Matrix |
| **Calibration** | CalibratedClassifierCV with isotonic regression |

## Repository Structure

This lab is part of the `MLOps_Labs` repository:

```
MLOps_Labs/                          (repository root)
├── .github/workflows/
│   ├── lab2_model_training.yml      # Triggered on push to main (github_lab2 changes)
│   └── lab2_model_calibration.yml   # Scheduled daily + manual trigger
├── fastapi_lab1/                    # Lab 1
├── github_lab2/                     # Lab 2 (this folder)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── train_model.py           # Training script
│   │   ├── evaluate_model.py        # Evaluation script
│   │   └── calibrate_model.py       # Calibration script
│   ├── data/                        # Test split (pickle files)
│   ├── models/                      # Trained models (.joblib)
│   ├── metrics/                     # Metrics JSON files
│   ├── requirements.txt
│   └── README.md
└── .gitignore
```

**Note:** GitHub Actions workflows must be in `.github/workflows/` at the repository root to be recognized by GitHub.

## Getting Started

### Prerequisites
- Python 3.9+
- Git

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranav1011/IE7374-MLOps.git
   cd IE7374-MLOps/github_lab2
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running Locally

1. **Train the model:**
   ```bash
   TS=$(date +%Y%m%d%H%M%S)
   python src/train_model.py --timestamp "$TS"
   ```

2. **Evaluate the model:**
   ```bash
   python src/evaluate_model.py --timestamp "$TS"
   ```

3. **Calibrate the model (optional):**
   ```bash
   python src/calibrate_model.py --timestamp "$TS" --method isotonic
   ```

4. **Check outputs:**
   ```bash
   ls -1 models/
   ls -1 metrics/
   ```

### Expected Outputs

After running the scripts, you should see:
- `models/model_{timestamp}_gb_model.joblib` - Trained GradientBoosting model
- `metrics/{timestamp}_metrics.json` - Evaluation metrics
- `models/model_{timestamp}_cal_isotonic.joblib` - Calibrated model (if calibration was run)
- `metrics/{timestamp}_calibration_isotonic.json` - Calibration metrics

## GitHub Actions Workflows

### 1. Model Training on Push (`lab2_model_training.yml`)

**Location:** `.github/workflows/lab2_model_training.yml` (at repo root)

**Trigger:** Push to `main` branch (only when `github_lab2/` files change)

**Steps:**
1. Checkout code
2. Set up Python 3.9
3. Install dependencies from `github_lab2/requirements.txt`
4. Generate timestamp for versioning
5. Train GradientBoostingClassifier on Breast Cancer dataset
6. Evaluate model and compute metrics
7. Store versioned model in `models/`
8. Commit and push metrics and model back to repository

### 2. Model Calibration (`lab2_model_calibration.yml`)

**Location:** `.github/workflows/lab2_model_calibration.yml` (at repo root)

**Trigger:** Daily at midnight UTC (cron) or manual dispatch via GitHub UI

**Steps:**
1. Checkout code
2. Set up Python 3.9
3. Find the most recent trained model
4. Apply isotonic calibration using CalibratedClassifierCV
5. Save calibrated model and calibration metrics
6. Commit and push changes

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions |
| **F1 (weighted)** | Harmonic mean of precision and recall, weighted by class support |
| **Precision** | Proportion of positive identifications that were correct |
| **Recall** | Proportion of actual positives that were identified correctly |
| **ROC-AUC** | Area under the ROC curve (probability ranking quality) |
| **Brier Score** | Mean squared error of probability predictions (calibration) |

## Calibration

Model calibration ensures that predicted probabilities match actual event frequencies. This is important for:
- Medical diagnosis (probability of malignancy)
- Risk assessment
- Decision-making under uncertainty

The calibration workflow uses **isotonic regression** which is non-parametric and can correct any monotonic distortion in the probabilities.

## How It Differs From Other Implementations

| Aspect | This Implementation |
|--------|---------------------|
| **Dataset** | Breast Cancer (binary, 30 features) |
| **Model** | GradientBoostingClassifier |
| **Extra Metrics** | ROC-AUC, Recall |
| **Calibration** | Full implementation with Brier score comparison |

## License

MIT License
