# FastAPI Wine Classification API

A REST API built with **FastAPI** that serves a machine learning model for Wine classification using the sklearn Wine dataset.

## Features

- **Health Check Endpoint** - Verify API status
- **Prediction Endpoint** - Classify wine based on 13 chemical features
- Pre-trained model using sklearn

## Directory Structure

```
fastapi_lab1/
├── src/
│   ├── main.py        # FastAPI application and endpoints
│   ├── data.py        # Data loading and splitting functions
│   ├── train.py       # Model training script
│   └── predict.py     # Prediction function
├── model/             # Saved model artifacts
├── assets/            # Screenshots
└── requirements.txt   # Python dependencies
```

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Navigate to the lab directory:
```bash
cd fastapi_lab1
```

2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI server:
```bash
cd src
python main.py
```

Or using uvicorn directly:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - returns API status |
| `/predict` | POST | Predict wine class from features |
| `/docs` | GET | Swagger UI documentation |

## Usage

### Health Check
```bash
curl http://localhost:8000/
```
Response: `{"status": "healthy"}`

### Predict Wine Class
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.0,
    "malic_acid": 2.0,
    "ash": 2.3,
    "alcalinity_of_ash": 17.0,
    "magnesium": 100.0,
    "total_phenols": 2.5,
    "flavanoids": 2.5,
    "nonflavanoid_phenols": 0.3,
    "proanthocyanins": 1.5,
    "color_intensity": 5.0,
    "hue": 1.0,
    "od280_od315_of_diluted_wines": 3.0,
    "proline": 1000.0
  }'
```
Response: `{"response": 0}`

## Input Features

| Feature | Description |
|---------|-------------|
| alcohol | Alcohol content |
| malic_acid | Malic acid |
| ash | Ash content |
| alcalinity_of_ash | Alcalinity of ash |
| magnesium | Magnesium |
| total_phenols | Total phenols |
| flavanoids | Flavanoids |
| nonflavanoid_phenols | Non-flavanoid phenols |
| proanthocyanins | Proanthocyanins |
| color_intensity | Color intensity |
| hue | Hue |
| od280_od315_of_diluted_wines | OD280/OD315 of diluted wines |
| proline | Proline |

## Demo

### Swagger UI - Endpoints
![Endpoints](assets/Endpoint%201.png)

### Predict Endpoint
![Predict](assets/Endpoint%202.png)

### Prediction Response
![Response](assets/Endpoint%203.png)

### Curl Response
![Curl](assets/Curl%20Response.png)
