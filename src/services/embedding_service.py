import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from services.ranking_service import (
    rank_scores
)

def build_embedding_model():

    print("\nLoading Embedding Model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cuda"
    )

    return model

if __name__ == "__main__":
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


def build_document_embeddings(
    model,
    raw_documents,
    show_progress_bar=True,
    convert_to_numpy=True
):

    document_embeddings = model.encode(
        raw_documents,
        show_progress_bar=True
    )

    print("Embeddings Ready")

    return document_embeddings


def retrieve_embedding(
    query,
    model,
    document_embeddings,
    doc_ids,
    top_k=10
):

    query_embedding = model.encode([query],
                                   convert_to_numpy=True)

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    return rank_scores(
        scores,
        doc_ids,
        top_k
    )


def get_embedding_scores(
    query,
    model,
    document_embeddings
):

    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    return scores