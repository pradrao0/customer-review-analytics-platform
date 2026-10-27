# Customer Review Analytics Platform

A Python-based Machine Learning application that analyses customer reviews using DistilBERT embeddings and Scikit-learn models. The application exposes its prediction pipeline through a FastAPI REST API and generates summary analytics from uploaded review datasets.

## Overview

The platform processes customer review data and generates predictions for:

- **Sentiment Analysis:** Classifies reviews as positive, negative, or neutral, according to the model's output labels.
- **Fake Review Detection:** Predicts whether reviews are likely to be deceptive or truthful.
- **Rating Prediction:** Generates predicted ratings for customer reviews.
- **Review Analytics:** Produces summary statistics and review-level prediction results.

## Features

- CSV file upload through a FastAPI endpoint.
- DistilBERT-based text embedding generation.
- Scikit-learn models for review classification and rating prediction.
- Modular components for embeddings, model inference, and analytics generation.
- JSON responses containing summary statistics and individual review predictions.

## Tech Stack

- **Programming Language:** Python
- **API Framework:** FastAPI
- **Machine Learning:** Scikit-learn
- **Text Embeddings:** DistilBERT
- **Data Processing:** Pandas, NumPy
- **Model Loading:** Joblib
- **Deep Learning Framework:** PyTorch
- **NLP Tools:** Hugging Face Transformers

## Project Structure

```text
customer-review-analytics-platform/
├── app/
│   ├── main.py
│   ├── embeddings.py
│   ├── models_loader.py
│   ├── predictor.py
│   ├── analytics.py
│   └── routers/
│       └── prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. A user uploads a CSV file containing customer reviews.
2. The API identifies a supported review-text column.
3. DistilBERT converts the review text into numerical embeddings.
4. Scikit-learn models generate sentiment, fake-review, and rating predictions.
5. The analytics component calculates summary statistics.
6. The API returns the summary and individual predictions as JSON.

## API Endpoint

### `POST /predict`

Accepts a CSV file containing customer review text and returns review-level predictions and summary analytics.

**Supported review-text column names:**

- `text`
- `review`
- `reviews`
- `comment`
- `comments`
- `feedback`

### Example Request

Once the application and required model files are configured, start the FastAPI server and open the interactive API documentation at `/docs`.

Example request using cURL:

```bash
curl -X POST \
  -F "file=@sample_reviews.csv" \
  http://127.0.0.1:8000/predict
```

### Response Format

The API returns a JSON object containing:

- `summary`: Aggregate review statistics, including sentiment counts, fake-review counts, and average predicted rating.
- `predictions`: Individual review records with their predicted sentiment, fake-review classification, and predicted rating.

## Installation and Model Setup

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Obtain the trained model artifacts and DistilBERT model/tokenizer files required by the application.
5. Place the files in the paths expected by `app/models_loader.py`.
6. Start the API using the command below, from the repository root, once all required files are available:

```bash
uvicorn app.main:app --reload
```

**Important:** The required model artifacts are not included in this repository. The model setup must be completed before the prediction endpoint can run successfully. Refer to `app/models_loader.py` for the exact expected file paths.

## Future Improvements

- Add automated tests for API requests and model predictions.
- Improve input validation and error handling.
- Document model training procedures and evaluation metrics.
- Add reproducible model setup instructions.
- Evaluate model performance using appropriate test datasets and metrics.

## Author

**Pradhyumna Rao**
