import numpy as np

from rank_bm25 import BM25Okapi

from services.ranking_service import (
    rank_scores
)


def build_bm25(processed_documents):

    print("\nBuilding BM25...")

    tokenized_docs = [
        doc.split()
        for doc in processed_documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    print("BM25 Ready")

    return bm25


def retrieve_bm25(
    query,
    bm25,
    doc_ids,
    preprocess_text,
    top_k=10
):

    query = preprocess_text(query)

    tokens = query.split()

    scores = bm25.get_scores(tokens)

    return rank_scores(
        scores,
        doc_ids,
        top_k
    )

def get_bm25_scores(
    query,
    bm25,
    preprocess_text
):

    query = preprocess_text(query)

    tokens = query.split()

    scores = bm25.get_scores(tokens)

    return np.array(scores)