# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.svm import SVC

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder


# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv("app/data/deceptive-opinion.csv")

print(df.head())
print(df.info())
print(df.shape)


# ============================================
# SELECT REQUIRED COLUMNS
# ============================================

df = df[["text", "deceptive"]]

print("\nDataset After Selecting Columns:\n")
print(df.head())


# ============================================
# REMOVE MISSING VALUES
# ============================================

print("\nRows Before Removing Missing Values:")
print(df.shape)

df = df.dropna()

print("\nRows After Removing Missing Values:")
print(df.shape)


# ============================================
# REMOVE DUPLICATES
# ============================================

print("\nRows Before Removing Duplicates:")
print(df.shape)

df = df.drop_duplicates()

print("\nRows After Removing Duplicates:")
print(df.shape)


# ============================================
# ENCODE LABELS
# ============================================

encoder = LabelEncoder()

df["deceptive"] = encoder.fit_transform(df["deceptive"])

print("\nEncoded Labels:")
print(df["deceptive"].value_counts())

print("\nLabel Mapping:")
for label, value in zip(
    encoder.classes_,
    encoder.transform(encoder.classes_)
):
    print(f"{label} -> {value}")


# ============================================
# FEATURES AND TARGET
# ============================================

X = df["text"]
y = df["deceptive"]


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training Reviews :", X_train.shape)
print("Testing Reviews  :", X_test.shape)
print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)


# ============================================
# LOAD DISTILBERT
# ============================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = AutoModel.from_pretrained(
    "distilbert-base-uncased"
)

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

print("DistilBERT Loaded Successfully!")
print("Using Device :", device)


# ============================================
# CREATE EMBEDDINGS
# ============================================

def get_embeddings(texts, batch_size=64):

    texts = texts.tolist()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            f"Processed {min(i + batch_size, len(texts))}/{len(texts)} reviews"
        )

    return np.vstack(all_embeddings)


# ============================================
# CREATE TRAIN EMBEDDINGS
# ============================================

print("\nCreating Training Embeddings...")

X_train_embeddings = get_embeddings(X_train)

print("\nTraining Embeddings Shape:")
print(X_train_embeddings.shape)


# ============================================
# CREATE TEST EMBEDDINGS
# ============================================

print("\nCreating Testing Embeddings...")

X_test_embeddings = get_embeddings(X_test)

print("\nTesting Embeddings Shape:")
print(X_test_embeddings.shape)


# ============================================
# TRAIN SVM CLASSIFIER
# ============================================

print("\n========== TRAINING SVM ==========\n")

svm_model = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_embeddings,
    y_train
)

print("SVM Model Trained Successfully!")


# ============================================
# MAKE PREDICTIONS
# ============================================

print("\n========== MAKING PREDICTIONS ==========\n")

predictions = svm_model.predict(
    X_test_embeddings
)

print("Predictions Completed!")


# ============================================
# PREDICTION PROBABILITIES
# ============================================

prediction_probabilities = svm_model.predict_proba(
    X_test_embeddings
)

print("\nFirst Five Prediction Probabilities:\n")
print(prediction_probabilities[:5])


# ============================================
# MODEL EVALUATION
# ============================================

print("\n========== MODEL EVALUATION ==========\n")

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================
# SAVE SVM MODEL
# ============================================

joblib.dump(
    svm_model,
    "app/models/fake_review_model.pkl"
)

print("\nSVM Model Saved Successfully!")


# ============================================
# SAVE LABEL ENCODER
# ============================================

joblib.dump(
    encoder,
    "app/models/fake_review_label_encoder.pkl"
)

print("Label Encoder Saved Successfully!")


# ============================================
# SAVE DISTILBERT MODEL
# ============================================

model.save_pretrained(
    "app/models/fake_review_distilbert"
)

print("DistilBERT Model Saved Successfully!")


# ============================================
# SAVE TOKENIZER
# ============================================

tokenizer.save_pretrained(
    "app/models/fake_review_tokenizer"
)

print("Tokenizer Saved Successfully!")


# ============================================
# PROJECT COMPLETED
# ============================================

print("\n===================================")
print("FAKE REVIEW DETECTION COMPLETED!")
print("===================================")


