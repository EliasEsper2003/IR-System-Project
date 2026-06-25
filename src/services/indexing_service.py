import os

from services.storage_service import (
    save_pickle,
    load_pickle,
    save_embeddings,
    load_embeddings
)

from services.tfidf_service import (
    build_tfidf
)

from services.bm25_service import (
    build_bm25
)

from services.embedding_service import (
    build_embedding_model,
    build_document_embeddings
)

from services.clustering_service import (
    build_clusters
)

from services.topic_detection_service import (
    build_topic_names,
    detect_topic
)


def build_indexes(
    processed_documents,
    raw_documents,
    dataset_name
):

    folder_name = (
        dataset_name
        .replace("/", "_")
        .replace("-", "_")
    )

    save_dir = (
        f"saved_indexes/{folder_name}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    tfidf_path = (
        f"{save_dir}/tfidf.pkl"
    )

    tfidf_matrix_path = (
        f"{save_dir}/tfidf_matrix.pkl"
    )

    bm25_path = (
        f"{save_dir}/bm25.pkl"
    )

    embeddings_path = (
        f"{save_dir}/embeddings.npy"
    )

    cluster_model_path = (
        f"{save_dir}/cluster_model.pkl"
    )

    cluster_labels_path = (
        f"{save_dir}/cluster_labels.npy"
    )

    if (
        os.path.exists(tfidf_path)
        and os.path.exists(tfidf_matrix_path)
        and os.path.exists(bm25_path)
        and os.path.exists(embeddings_path)
        and os.path.exists(cluster_model_path)
        and os.path.exists(cluster_labels_path)
    ):

        print("Loading Saved Indexes...")

        tfidf_vectorizer = load_pickle(
            tfidf_path
        )

        tfidf_matrix = load_pickle(
            tfidf_matrix_path
        )

        bm25 = load_pickle(
            bm25_path
        )

        document_embeddings = load_embeddings(
            embeddings_path
        )

        cluster_model = load_pickle(
            cluster_model_path
        )

        cluster_labels = load_embeddings(
            cluster_labels_path
        )

        topic_names = build_topic_names(
            processed_documents,
            cluster_labels
        )

        model = build_embedding_model()

    
    else:
        print("Building Indexes...")


        tfidf_vectorizer, tfidf_matrix = (
            build_tfidf(
                processed_documents
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

        cluster_model, cluster_labels = (
            build_clusters(
                document_embeddings
            )
        )

        topic_names = build_topic_names(
            processed_documents,
            cluster_labels
        )

        save_pickle(
            tfidf_vectorizer,
            tfidf_path
        )

        save_pickle(
            tfidf_matrix,
            tfidf_matrix_path
        )

        save_pickle(
            bm25,
            bm25_path
        )

        save_embeddings(
            document_embeddings,
            embeddings_path
        )

        save_pickle(
            cluster_model,
            cluster_model_path
        )

        save_embeddings(
            cluster_labels,
            cluster_labels_path
        )

        print("Indexes Saved")

        


    print("Indexes Ready")


    return {

        "tfidf_vectorizer":
            tfidf_vectorizer,

        "tfidf_matrix":
            tfidf_matrix,

        "bm25":
            bm25,

        "model":
            model,

        "document_embeddings":
            document_embeddings,

        "cluster_model":
            cluster_model,

        "cluster_labels":
            cluster_labels,
        
        "topic_names":
            topic_names
    }
