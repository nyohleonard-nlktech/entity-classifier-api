# Entity Classifier API

**A FastAPI service for classifying text with a supervised FastText model.**

This repository loads a trained FastText model and exposes single-text and batch prediction endpoints. Predictions include the original input, predicted label, confidence score, and inference source.

## Features

- Loads the model once during application startup.
- Health endpoint that reports model readiness.
- Single prediction endpoint.
- Batch prediction endpoint.
- Basic whitespace/newline normalization before inference.
- Pydantic request and response models.

## Technology

- Python
- FastAPI and Pydantic
- FastText
- A serialized supervised model stored as `classifier_model.ftz`

## Requirements

Use a Python version compatible with the FastText wheel and NumPy version in `requirements.txt`. The current requirements file lists `fasttext-wheel` and constrains NumPy below 2.0.

**Dependency note:** `main.py` imports FastAPI and Pydantic, but the current `requirements.txt` does not list FastAPI, Uvicorn, or Pydantic explicitly. Install those API runtime packages or update the dependency file before running the API.

## Installation

```bash
git clone https://github.com/nyohleonard-nlktech/entity-classifier-api.git
cd entity-classifier-api

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install declared dependencies and the API runtime:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn pydantic
```

## Run

Ensure `classifier_model.ftz` is present at the path configured in `config.py`, then start the service:

```bash
uvicorn main:app --reload
```

- Base URL: `http://127.0.0.1:8000`
- Health: `GET /health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

## API reference

### Single prediction

`POST /predict`

```json
{
  "text": "Example text to classify"
}
```

### Batch prediction

`POST /predict/batch`

```json
{
  "texts": [
    "First example",
    "Second example"
  ]
}
```

The batch response is a list of classification results. The exact label set is determined by the trained model.

## Project layout

```text
.
├── main.py                 # FastAPI application and routes
├── app.py                  # Additional application entry point
├── engine.py               # FastText inference wrapper
├── config.py               # Model path configuration
├── classifier_model.ftz    # Trained model artifact
├── Dockerfile
├── requirements.txt
└── .devcontainer/
```

## Model and evaluation notes

- The label vocabulary and training data are model-dependent; inspect the training pipeline before retraining or changing labels.
- Confidence values are FastText model scores and should be calibrated/evaluated before high-impact use.
- Add tests for empty input, unknown labels, batch size limits, malformed payloads, and missing model files.
- Do not send confidential text to a hosted deployment without an appropriate privacy policy.

## License

No license file is currently present. All rights are reserved unless an explicit license is added.
