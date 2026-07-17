from fastapi import APIRouter, UploadFile, File
import pandas as pd

from app.embeddings import get_embeddings
from app.predictor import predict_reviews
from app.analytics import generate_summary

router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    print("===== PREDICT ENDPOINT HIT =====")

    try:
        df = pd.read_csv(file.file)
        print("CSV READ SUCCESSFULLY")
    except Exception as e:
        print("CSV ERROR:", e)
        return {
            "error": str(e)
        }
 

    possible_columns = [
        "text",
        "review",
        "reviews",
        "comment",
        "comments",
        "feedback"
    ]

    review_column = None

    for column in possible_columns:
        if column in df.columns:
            review_column = column
            break

    if review_column is None:
        return {
            "error": (
                "CSV must contain one of these columns: "
                "text, review, reviews, comment, comments, feedback"
            )
        }

    # ============================================
    # RENAME REVIEW COLUMN
    # ============================================

    df = df.rename(columns={review_column: "text"})

    # ============================================
    # CREATE DISTILBERT EMBEDDINGS
    # ============================================

    embeddings = get_embeddings(df["text"])

    # ============================================
    # RUN ALL ML MODELS
    # ============================================

    predictions = predict_reviews(embeddings)

    # ============================================
    # STORE PREDICTIONS IN DATAFRAME
    # ============================================

    df["Fake Review"] = predictions["fake_reviews"]

    df["Sentiment"] = predictions["sentiments"]

    df["Predicted Rating"] = predictions["ratings"].round(1)

    # ============================================
    # GENERATE ANALYTICS REPORT
    # ============================================

    summary = generate_summary(df)

    # ============================================
    # RETURN REPORT
    # ============================================

    return {
    "summary": summary,
    "predictions": df.to_dict(orient="records")
    }