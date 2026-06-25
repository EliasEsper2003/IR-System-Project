import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.ranking_service import (
    rank_scores
)


def build_tfidf(processed_documents):

    print("\nBuilding TF-IDF...")

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        processed_documents
    )

    print("TF-IDF Ready")

    return vectorizer, tfidf_matrix


def get_tfidf_scores(
    query,
    vectorizer,
    tfidf_matrix,
    preprocess_text
):

    query = preprocess_text(query)

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    return scores


def retrieve_tfidf(
    query,
    vectorizer,
    tfidf_matrix,
    doc_ids,
    preprocess_text,
    top_k=10
):

    scores = get_tfidf_scores(
        query,
        vectorizer,
        tfidf_matrix,
        preprocess_text
    )

    return rank_scores(
        scores,
        doc_ids,
        top_k
    )