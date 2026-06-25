import numpy as np

from services.evaluation_service import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    average_precision
)

from services.retrieval_service import (
    retrieve_all_models
)


def run_experiment(context):

    queries = context["queries"]
    qrel_dict = context["qrel_dict"]

    refine_query = context["refine_query"]

    use_query_refinement = context["use_query_refinement"]

    tfidf_vectorizer = context["tfidf_vectorizer"]
    tfidf_matrix = context["tfidf_matrix"]
    

    bm25 = context["bm25"]

    model = context["model"]
    document_embeddings = context["document_embeddings"]

    doc_ids = context["doc_ids"]

    preprocess_text = context["preprocess_text"]

    get_tfidf_scores = context["get_tfidf_scores"]
    get_bm25_scores = context["get_bm25_scores"]
    get_embedding_scores = context["get_embedding_scores"]

    normalize = context["normalize"]

    test_queries = queries

    tfidf_recalls = []
    tfidf_precisions = []
    tfidf_ndcgs = []
    tfidf_maps = []

    bm25_recalls = []
    bm25_precisions = []
    bm25_ndcgs = []
    bm25_maps = []

    embedding_recalls = []
    embedding_precisions = []
    embedding_ndcgs = []
    embedding_maps = []

    hybrid_recalls = []
    hybrid_precisions = []
    hybrid_ndcgs = []
    hybrid_maps = []

    serial_recalls = []
    serial_precisions = []
    serial_ndcgs = []
    serial_maps = []

    for q in test_queries:

        if q.query_id not in qrel_dict:
            continue

        relevant_docs = qrel_dict[q.query_id]

        if use_query_refinement:
            current_query = refine_query(q.text)
        else:
            current_query = q.text

        retrieval_results = retrieve_all_models(
            current_query,
            context
        )

        if len(relevant_docs) == 0:
            continue

        # TF-IDF

        retrieved = retrieval_results["tfidf"]

        tfidf_recalls.append(
            recall_at_k(retrieved, relevant_docs)
        )

        tfidf_precisions.append(
            precision_at_k(retrieved, relevant_docs)
        )

        tfidf_ndcgs.append(
            ndcg_at_k(retrieved, relevant_docs)
        )

        tfidf_maps.append(
            average_precision(retrieved, relevant_docs)
        )

        # BM25

        retrieved = retrieval_results["bm25"]

        bm25_recalls.append(
            recall_at_k(retrieved, relevant_docs)
        )

        bm25_precisions.append(
            precision_at_k(retrieved, relevant_docs)
        )

        bm25_ndcgs.append(
            ndcg_at_k(retrieved, relevant_docs)
        )

        bm25_maps.append(
            average_precision(retrieved, relevant_docs)
        )

        # Embedding

        retrieved = retrieval_results["embedding"]

        embedding_recalls.append(
            recall_at_k(retrieved, relevant_docs)
        )

        embedding_precisions.append(
            precision_at_k(retrieved, relevant_docs)
        )

        embedding_ndcgs.append(
            ndcg_at_k(retrieved, relevant_docs)
        )

        embedding_maps.append(
            average_precision(retrieved, relevant_docs)
        )

        # Hybrid

        retrieved = retrieval_results["hybrid"]

        hybrid_recalls.append(
            recall_at_k(retrieved, relevant_docs)
        )

        hybrid_precisions.append(
            precision_at_k(retrieved, relevant_docs)
        )

        hybrid_ndcgs.append(
            ndcg_at_k(retrieved, relevant_docs)
        )

        hybrid_maps.append(
            average_precision(retrieved, relevant_docs)
        )

        # Hybrid Serial

        retrieved = retrieval_results["serial"]

        serial_recalls.append(
            recall_at_k(retrieved, relevant_docs)
        )

        serial_precisions.append(
            precision_at_k(retrieved, relevant_docs)
        )

        serial_ndcgs.append(
            ndcg_at_k(retrieved, relevant_docs)
        )

        serial_maps.append(
            average_precision(retrieved, relevant_docs)
        )

    return {
        "tfidf_recalls": tfidf_recalls,
        "tfidf_precisions": tfidf_precisions,
        "tfidf_maps": tfidf_maps,
        "tfidf_ndcgs": tfidf_ndcgs,

        "bm25_recalls": bm25_recalls,
        "bm25_precisions": bm25_precisions,
        "bm25_maps": bm25_maps,
        "bm25_ndcgs": bm25_ndcgs,

        "embedding_recalls": embedding_recalls,
        "embedding_precisions": embedding_precisions,
        "embedding_maps": embedding_maps,
        "embedding_ndcgs": embedding_ndcgs,

        "hybrid_recalls": hybrid_recalls,
        "hybrid_precisions": hybrid_precisions,
        "hybrid_maps": hybrid_maps,
        "hybrid_ndcgs": hybrid_ndcgs,

        "serial_recalls": serial_recalls,
        "serial_precisions": serial_precisions,
        "serial_maps": serial_maps,
        "serial_ndcgs": serial_ndcgs
    }