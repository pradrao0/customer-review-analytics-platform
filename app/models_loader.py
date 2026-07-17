import joblib
from transformers import AutoTokenizer, AutoModel


def load_models():

    print("Loading Models...")

    # Fake Review Model
    fake_review_model = joblib.load(
        "app/models/fake_review_model.pkl"
    )

    fake_review_label_encoder = joblib.load(
        "app/models/fake_review_label_encoder.pkl"
    )

    # Sentiment Model
    sentiment_model = joblib.load(
        "app/models/sentiment_model.pkl"
    )

    # Rating Model
    rating_model = joblib.load(
        "app/models/rating_model.pkl"
    )

    # DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(
        "app/models/distilbert_tokenizer"
    )

    distilbert = AutoModel.from_pretrained(
        "app/models/distilbert_model"
    )

    distilbert.eval()

    print("Models Loaded Successfully!")

    return (
        fake_review_model,
        fake_review_label_encoder,
        sentiment_model,
        rating_model,
        tokenizer,
        distilbert
    )


(
    fake_review_model,
    fake_review_label_encoder,
    sentiment_model,
    rating_model,
    tokenizer,
    distilbert
) = load_models()