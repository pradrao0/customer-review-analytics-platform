# ============================================
# IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np

# Train-Test Split
from sklearn.model_selection import train_test_split

# Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor

# Hugging Face Transformer
from transformers import AutoTokenizer, AutoModel
# PyTorch
import torch

# Save and Load Model
import joblib


import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("Current Working Directory:")
print(os.getcwd())

print("\nCurrent Script:")
print(__file__)

# Now load the dataset
df = pd.read_csv("app/data/rating_dataset.csv")


print("first five rows are ")
print(df.head())
print("last five rows are")
print(df.tail())
print("information about our dataset")
print(df.shape)
print("dataset information is here")
print(df.info())
print("columns")
print(df.columns)

# ============================================
# REMOVE UNNECESSARY COLUMNS
# ============================================

df = df.drop(
    columns=[
        "Id",
        "ProductId",
        "UserId",
        "ProfileName",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
        "Time"
    ]
)

print("\n========== DATASET AFTER REMOVING COLUMNS ==========\n")
print(df.head())

print("\nColumns Remaining:")
print(df.columns)


# ============================================
# REMOVE MISSING VALUES
# ============================================

print("\nRows Before Removing Missing Values:")
print(df.shape)

df = df.dropna()

print("\nRows After Removing Missing Values:")
print(df.shape)


# ============================================
# REMOVE DUPLICATE ROWS
# ============================================

print("\nRows Before Removing Duplicates:")
print(df.shape)

df = df.drop_duplicates()

print("\nRows After Removing Duplicates:")
print(df.shape)

# ============================================
# USE ONLY 10,000 REVIEWS
# ============================================

df = df.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

print("\nRows After Sampling:")
print(df.shape)

# ============================================
# COMBINE SUMMARY AND TEXT
# ============================================

df["Review"] = df["Summary"] + " " + df["Text"]

print("\nFirst Combined Review:\n")
print(df["Review"].iloc[0])


# ============================================
# FEATURES AND TARGET
# ============================================

X = df["Review"]
y = df["Score"]


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
print("Training Ratings :", y_train.shape)
print("Testing Ratings  :", y_test.shape)


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
# TRAIN RANDOM FOREST REGRESSOR
# ============================================

print("\n========== TRAINING RANDOM FOREST ==========\n")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_embeddings,
    y_train
)

print("Random Forest Model Trained Successfully!")


# ============================================
# MAKE PREDICTIONS
# ============================================

print("\n========== PREDICTING RATINGS ==========\n")

predictions = rf_model.predict(
    X_test_embeddings
)

print("Prediction Completed!")


# ============================================
# EVALUATE MODEL
# ============================================

print("\n========== MODEL EVALUATION ==========\n")

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


# ============================================
# SAVE RANDOM FOREST MODEL
# ============================================

joblib.dump(
    rf_model,
    "app/models/rating_model.pkl"
)

print("\nRandom Forest Model Saved Successfully!")


# ============================================
# SAVE DISTILBERT MODEL
# ============================================

model.save_pretrained(
    "app/models/distilbert_model"
)

print("DistilBERT Model Saved Successfully!")


# ============================================
# SAVE TOKENIZER
# ============================================

tokenizer.save_pretrained(
    "app/models/distilbert_tokenizer"
)

print("Tokenizer Saved Successfully!")


# ============================================
# PROJECT COMPLETED
# ============================================

print("\n===================================")
print("RATING PREDICTION MODEL COMPLETED!")
print("===================================")