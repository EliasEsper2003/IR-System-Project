import os
os.environ["PYTHONUTF8"] = "1"

import numpy as np
import ir_datasets

from services.tfidf_service import (
    build_tfidf,
    retrieve_tfidf,
    get_tfidf_scores
)

from services.bm25_service import (
    build_bm25,
    retrieve_bm25,
    get_bm25_scores
)

from services.embedding_service import (
    build_embedding_model,
    build_document_embeddings,
    retrieve_embedding,
    get_embedding_scores
)

from services.evaluation_service import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    average_precision,
    normalize
)

from services.hybrid_service import (
    retrieve_hybrid,
    retrieve_hybrid_serial
)

from services.dataset_service import (
    load_dataset_data,
    load_queries_qrels
)

from services.preprocessing_service import (
    preprocess_text
)

from services.experiment_service import run_experiment

from services.report_service import print_results

from services.config import (
    DATASET_NAME,
    MAX_DOCS,
    TOP_K,
    USE_QUERY_REFINEMENT
)

from services.query_refinement_service import (
    refine_query
)


(
    dataset,
    raw_documents,
    processed_documents,
    doc_ids
) = load_dataset_data(
    DATASET_NAME,
    MAX_DOCS,
    preprocess_text
)

tfidf_vectorizer, tfidf_matrix = build_tfidf(
    processed_documents
)


queries, qrel_dict = (
    load_queries_qrels(
        dataset,
        doc_ids
    )
)

bm25 = build_bm25(
    processed_documents
)

model = build_embedding_model()

document_embeddings = (
    build_document_embeddings(
        model,
        raw_documents
    )
)

experiment_context = {
    "queries": queries,
    "qrel_dict": qrel_dict,

    "tfidf_vectorizer": tfidf_vectorizer,
    "tfidf_matrix": tfidf_matrix,

    "bm25": bm25,

    "model": model,
    "document_embeddings": document_embeddings,

    "doc_ids": doc_ids,

    "preprocess_text": preprocess_text,

    "get_tfidf_scores": get_tfidf_scores,
    "get_bm25_scores": get_bm25_scores,
    "get_embedding_scores": get_embedding_scores,

    "normalize": normalize,

    "refine_query": refine_query,

    "use_query_refinement": USE_QUERY_REFINEMENT,
}

results = run_experiment(
    experiment_context
)

print_results(results)