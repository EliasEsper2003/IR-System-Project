import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_hybrid(
    query,
    doc_ids,
    bm25,
    model,
    document_embeddings,
    preprocess_text,
    get_bm25_scores,
    get_embedding_scores,
    normalize,
    bm25_weight=0.7,
    embedding_weight=0.3,
    top_k=10
):

    bm25_scores = normalize(
        get_bm25_scores(
            query,
            bm25,
            preprocess_text
        )
    )

    embedding_scores = normalize(
        get_embedding_scores(
            query,
            model,
            document_embeddings
        )
    )

    hybrid_scores = (
        bm25_weight * bm25_scores +
        embedding_weight * embedding_scores
    )

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    return [
        doc_ids[idx]
        for idx in top_indices
    ]

def retrieve_hybrid_serial(
    query,
    doc_ids,
    bm25,
    model,
    document_embeddings,
    preprocess_text,
    get_bm25_scores,
    top_k=10
):

    query_processed = preprocess_text(query)

    bm25_scores = bm25.get_scores(
        query_processed.split()
    )

    candidate_indices = np.argsort(
        bm25_scores
    )[::-1][:100]

    query_embedding = model.encode([query])

    candidate_embeddings = (
        document_embeddings[candidate_indices]
    )

    embedding_scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    reranked_indices = np.argsort(
        embedding_scores
    )[::-1][:top_k]

    final_doc_ids = [
        doc_ids[candidate_indices[idx]]
        for idx in reranked_indices
    ]

    return final_doc_ids