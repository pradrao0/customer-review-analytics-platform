import pandas as pd


def generate_summary(df):

    total_reviews = int(len(df))

    positive_reviews = int(
        (df["Sentiment"] == "positive").sum()
    )

    negative_reviews = int(
        (df["Sentiment"] == "negative").sum()
    )

    neutral_reviews = int(
        (df["Sentiment"] == "neutral").sum()
    )

    fake_reviews = int(
        (df["Fake Review"] == "deceptive").sum()
    )

    genuine_reviews = int(
        (df["Fake Review"] == "truthful").sum()
    )

    average_rating = float(
        round(
            df["Predicted Rating"].mean(),
            2
        )
    )

    return {

        "Total Reviews": total_reviews,

        "Sentiment": {

            "Positive": positive_reviews,
            "Negative": negative_reviews,
            "Neutral": neutral_reviews

        },

        "Fake Reviews": {

            "Likely Fake": fake_reviews,
            "Likely Genuine": genuine_reviews

        },

        "Average Rating": average_rating

    }