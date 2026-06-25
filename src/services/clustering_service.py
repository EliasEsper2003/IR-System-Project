from sklearn.cluster import MiniBatchKMeans
import numpy as np


def build_clusters(
    document_embeddings,
    n_clusters=20
):

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(
        document_embeddings
    )

    return model, labels


def predict_query_cluster(
    query_embedding,
    cluster_model
):

    return cluster_model.predict(
        query_embedding
    )[0]


def get_cluster_document_indices(
    cluster_id,
    cluster_labels
):

    return np.where(
        cluster_labels == cluster_id
    )[0]

TOPIC_NAMES = {
    0: "Topic 0",
    1: "Topic 1",
    2: "Topic 2",
    3: "Topic 3",
    4: "Topic 4",
    5: "Topic 5",
    6: "Topic 6",
    7: "Topic 7",
    8: "Topic 8",
    9: "Topic 9",
    10: "Topic 10",
    11: "Topic 11",
    12: "Topic 12",
    13: "Topic 13",
    14: "Topic 14",
    15: "Topic 15",
    16: "Topic 16",
    17: "Topic 17",
    18: "Topic 18",
    19: "Topic 19"
}


def get_topic_name(cluster_id):

    return TOPIC_NAMES.get(
        cluster_id,
        "Unknown Topic"
    )