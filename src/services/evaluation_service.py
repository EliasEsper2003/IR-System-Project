import numpy as np


def recall_at_k(retrieved, relevant, k=10):

    retrieved = retrieved[:k]

    hits = len(set(retrieved) & relevant)

    return hits / len(relevant)


def precision_at_k(retrieved, relevant, k=10):

    retrieved = retrieved[:k]

    hits = len(set(retrieved) & relevant)

    return hits / k


def ndcg_at_k(retrieved, relevant, k=10):

    retrieved = retrieved[:k]

    dcg = 0

    for i, doc_id in enumerate(retrieved):

        if doc_id in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    if idcg == 0:
        return 0

    return dcg / idcg


def average_precision(retrieved, relevant, k=10):

    retrieved = retrieved[:k]

    hits = 0
    ap = 0

    for i, doc_id in enumerate(retrieved):

        if doc_id in relevant:

            hits += 1

            ap += hits / (i + 1)

    if len(relevant) == 0:
        return 0

    return ap / len(relevant)


def normalize(scores):

    return (
        scores - scores.min()
    ) / (
        scores.max() - scores.min() + 1e-9
    )