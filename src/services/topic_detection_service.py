from collections import Counter


def detect_topic(
    query,
    embedding_model,
    cluster_model,
    topic_names
):

    query_embedding = (
        embedding_model.encode([query])
    )

    cluster_id = cluster_model.predict(
        query_embedding
    )[0]

    topic_name = topic_names.get(
        cluster_id,
        f"Topic {cluster_id}"
    )

    return cluster_id, topic_name


def build_topic_names(
    processed_documents,
    cluster_labels,
    top_n_words=5
):

    ignore_words = {
        "one",
        "two",
        "three",
        "would",
        "could",
        "also",
        "like",
        "get",
        "got",
        "going",
        "go",
        "say",
        "said",
        "make",
        "made",
        "many",
        "much",
        "people",
        "using",
        "used",
        "use"
    }

    topic_names = {}

    for cluster_id in set(cluster_labels):

        cluster_docs = [
            processed_documents[i]
            for i in range(len(processed_documents))
            if cluster_labels[i] == cluster_id
        ]

        words = []

        for doc in cluster_docs[:200]:

            words.extend(
                [
                    word
                    for word in doc.split()
                    if len(word) > 3
                    and word not in ignore_words
                ]
            )

        common_words = Counter(words).most_common(
            top_n_words
        )

        topic_names[cluster_id] = ", ".join(
            [word for word, _ in common_words]
        )

    return topic_names