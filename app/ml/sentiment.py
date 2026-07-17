import pandas as pd
import numpy as np
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import AutoTokenizer, AutoModel

# -------------------------
# LOAD DATASET
# -------------------------

df = pd.read_csv("app/data/sentiment_dataset.csv")

texts = df["text"]
labels = df["label"]

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)

# -------------------------
# LOAD TOKENIZER & TRANSFORMER
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

transformer = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

# -------------------------
# FUNCTION TO CREATE EMBEDDINGS
# -------------------------

def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = transformer(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.squeeze().numpy()

# -------------------------
# CREATE TRAIN EMBEDDINGS
# -------------------------

X_train_embeddings = []

for text in X_train:

    embedding = get_embedding(text)

    X_train_embeddings.append(embedding)

X_train_embeddings = np.array(X_train_embeddings)

# -------------------------
# CREATE TEST EMBEDDINGS
# -------------------------

X_test_embeddings = []

for text in X_test:

    embedding = get_embedding(text)

    X_test_embeddings.append(embedding)

X_test_embeddings = np.array(X_test_embeddings)

print("Train Embeddings Shape:", X_train_embeddings.shape)
print("Test Embeddings Shape:", X_test_embeddings.shape)

# -------------------------
# CREATE MODEL
# -------------------------

model = LogisticRegression(
    max_iter=1000
)

# -------------------------
# TRAIN MODEL
# -------------------------

model.fit(
    X_train_embeddings,
    y_train
)

# -------------------------
# MAKE PREDICTIONS
# -------------------------

y_pred = model.predict(
    X_test_embeddings
)

# -------------------------
# EVALUATE MODEL
# -------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print()
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# -------------------------
# SAVE MODEL
# -------------------------

joblib.dump(
    model,
    "app/models/sentiment_model.pkl"
)

print()
print("Model Saved Successfully")

# -------------------------
# TEST CUSTOM REVIEWS
# -------------------------

review_1 = "This product is amazing"

review_2 = "Worst purchase ever"

embedding_1 = get_embedding(review_1)

embedding_2 = get_embedding(review_2)

prediction_1 = model.predict([embedding_1])

prediction_2 = model.predict([embedding_2])

print()
print("Review 1 Prediction:", prediction_1[0])

print("Review 2 Prediction:", prediction_2[0])