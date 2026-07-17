import numpy as np
import torch

from app.models_loader import (
    tokenizer,
    distilbert
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

distilbert.to(device)


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

            outputs = distilbert(**tokens)

        embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

    return np.vstack(all_embeddings)