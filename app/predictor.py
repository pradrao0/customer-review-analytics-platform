from app.models_loader import (
    fake_review_model,
    fake_review_label_encoder,
    sentiment_model,
    rating_model
)


def predict_reviews(embeddings):

    # ============================================
    # FAKE REVIEW PREDICTION
    # ============================================

    fake_predictions = fake_review_model.predict(
        embeddings
    )

    fake_predictions = (
        fake_review_label_encoder.inverse_transform(
            fake_predictions
        )
    )

    # ============================================
    # SENTIMENT PREDICTION
    # ============================================

    sentiment_predictions = sentiment_model.predict(
        embeddings
    )

    # ============================================
    # RATING PREDICTION
    # ============================================

    rating_predictions = rating_model.predict(
        embeddings
    )

    return {
        "fake_reviews": fake_predictions,
        "sentiments": sentiment_predictions,
        "ratings": rating_predictions
    }