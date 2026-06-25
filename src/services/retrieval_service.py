from services.tfidf_service import retrieve_tfidf

from services.bm25_service import retrieve_bm25

from services.embedding_service import (
    retrieve_embedding
)

from services.hybrid_service import (
    retrieve_hybrid,
    retrieve_hybrid_serial
)


def retrieve_all_models(
    query,
    context
):

    tfidf_results = retrieve_tfidf(
        query,
        context["tfidf_vectorizer"],
        context["tfidf_matrix"],
        context["doc_ids"],
        context["preprocess_text"]
    )

    bm25_results = retrieve_bm25(
        query,
        context["bm25"],
        context["doc_ids"],
        context["preprocess_text"]
    )

    embedding_results = retrieve_embedding(
        query,
        context["model"],
        context["document_embeddings"],
        context["doc_ids"]
    )

    hybrid_results = retrieve_hybrid(
        query,
        context["doc_ids"],
        context["bm25"],
        context["model"],
        context["document_embeddings"],
        context["preprocess_text"],
        context["get_bm25_scores"],
        context["get_embedding_scores"],
        context["normalize"]
    )

    serial_results = retrieve_hybrid_serial(
        query,
        context["doc_ids"],
        context["bm25"],
        context["model"],
        context["document_embeddings"],
        context["preprocess_text"],
        context["get_bm25_scores"]
    )

    return {
        "tfidf": tfidf_results,
        "bm25": bm25_results,
        "embedding": embedding_results,
        "hybrid": hybrid_results,
        "serial": serial_results
    }