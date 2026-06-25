import numpy as np


def rank_scores(
    scores,
    doc_ids,
    top_k=10
):

    ranked_indices = (
        np.argsort(scores)[::-1]
    )

    ranked_indices = (
        ranked_indices[:top_k]
    )

    return [
        doc_ids[i]
        for i in ranked_indices
    ]