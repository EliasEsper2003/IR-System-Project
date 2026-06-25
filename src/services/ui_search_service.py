from services.query_refinement_service import (
    refine_query
)


from services.bm25_service import (
    retrieve_bm25,
    get_bm25_scores
)

from services.embedding_service import (
    retrieve_embedding,
    get_embedding_scores
)

from services.evaluation_service import normalize

from services.clustering_service import (
    predict_query_cluster,
    get_cluster_document_indices
)

import streamlit as st

def search_documents(
    query,
    model_name,
    indexes,
    doc_ids,
    preprocess_text,
    top_k,
    use_query_refinement,
    use_clustering,
    bm25_weight=0.7,
    embedding_weight=0.3
):

    if use_query_refinement:

        query = refine_query(query)


    tfidf_vectorizer = indexes["tfidf_vectorizer"]
    tfidf_matrix = indexes["tfidf_matrix"]

    bm25 = indexes["bm25"]

    embedding_model = indexes["model"]
    document_embeddings = indexes["document_embeddings"]

    cluster_model = indexes["cluster_model"]
    cluster_labels = indexes["cluster_labels"]

    if use_clustering:

        query_embedding = embedding_model.encode(
            [query]
        )

        cluster_id = predict_query_cluster(
            query_embedding,
            cluster_model
        )

        cluster_doc_indices = (
            get_cluster_document_indices(
                cluster_id,
                cluster_labels
            )
        )

        filtered_doc_ids = [
            doc_ids[i]
            for i in cluster_doc_indices
        ]

        filtered_embeddings = (
            document_embeddings[
                cluster_doc_indices
            ]
        )

        st.write(
            f"Documents in Cluster: {len(cluster_doc_indices)}"
        )

        print(
            f"Selected Cluster: {cluster_id}"
        )

    

    from services.tfidf_service import retrieve_tfidf
    from services.bm25_service import retrieve_bm25
    from services.embedding_service import retrieve_embedding
    from services.hybrid_service import (
        retrieve_hybrid,
        retrieve_hybrid_serial
    )


    if model_name == "TF-IDF":

        if use_clustering:

            filtered_tfidf_matrix = (
                tfidf_matrix[
                    cluster_doc_indices
                ]
            )

            return retrieve_tfidf(
                query,
                tfidf_vectorizer,
                filtered_tfidf_matrix,
                filtered_doc_ids,
                preprocess_text,
                top_k
            )

        return retrieve_tfidf(
            query,
            tfidf_vectorizer,
            tfidf_matrix,
            doc_ids,
            preprocess_text,
            top_k
        )


    elif model_name == "BM25":

        return retrieve_bm25(
            query,
            bm25,
            doc_ids,
            preprocess_text,
            top_k
        )


    elif model_name == "Embedding":

        if use_clustering:

            return retrieve_embedding(
                query,
                embedding_model,
                filtered_embeddings,
                filtered_doc_ids,
                top_k
            )

        return retrieve_embedding(
            query,
            embedding_model,
            document_embeddings,
            doc_ids,
            top_k
        )


    elif model_name == "Hybrid":

        return retrieve_hybrid(
            query,
            doc_ids,
            bm25,
            embedding_model,
            document_embeddings,
            preprocess_text,
            get_bm25_scores,
            get_embedding_scores,
            normalize,
            bm25_weight,
            embedding_weight,
            top_k
        )


    elif model_name == "Hybrid Serial":

        return retrieve_hybrid_serial(
            query,
            doc_ids,
            bm25,
            embedding_model,
            document_embeddings,
            preprocess_text,
            get_bm25_scores,
            top_k
        )

    return []