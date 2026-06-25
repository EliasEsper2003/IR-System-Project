import streamlit as st
import pandas as pd


from services.dataset_service import load_dataset_data
from services.preprocessing_service import preprocess_text

from services.indexing_service import build_indexes

from services.ui_search_service import search_documents

from services.config import (
    DATASET_NAME,
)

from services.clustering_service import (
    predict_query_cluster,
    get_cluster_document_indices
)

from services.database_service import (
    get_document
)

from services.topic_detection_service import (
    detect_topic
)

from services.report_service import (
    build_results_dataframe
)

from services.experiment_service import (
    run_experiment
)

from services.dataset_service import (
    load_queries_qrels
)

from services.evaluation_service import (
    normalize
)

from services.tfidf_service import (
    get_tfidf_scores
)

from services.bm25_service import (
    get_bm25_scores
)

from services.embedding_service import (
    get_embedding_scores
)

from services.query_refinement_service import (
    refine_query
)

st.set_page_config(
    page_title="IR Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Information Retrieval System")

st.write(
    "Search using TF-IDF, BM25, Embedding and Hybrid Models"
)


@st.cache_resource
def load_system(
    dataset_name
):

    dataset, raw_documents, processed_documents, doc_ids = (
        load_dataset_data(
            dataset_name,
            preprocess_text,
        )
    )

    indexes = build_indexes(
        processed_documents,
        raw_documents,
        dataset_name
    )

    return (
        dataset,
        raw_documents,
        processed_documents,
        doc_ids,
        indexes
    )


dataset_name = st.selectbox(
    "Choose Dataset",
    [
        "beir/webis-touche2020/v2",
        "beir/trec-covid"
    ]
)


(
    dataset,
    raw_documents,
    processed_documents,
    doc_ids,
    indexes
) = load_system(
    dataset_name
)

cluster_labels = indexes["cluster_labels"]

cluster_counts = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

st.subheader(
    "Topic Distribution Across Documents"
)

st.bar_chart(
    cluster_counts
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Number of Clusters",
        len(cluster_counts)
    )

with col2:
    st.metric(
        "Total Documents",
        len(cluster_labels)
    )

st.info(
    f"Current Dataset: {dataset_name}"
 )

model_name = st.selectbox(
    "Choose Model",
    [
        "TF-IDF",
        "BM25",
        "Embedding",
        "Hybrid",
        "Hybrid Serial"
    ]
)

bm25_weight = 0.7
embedding_weight = 0.3

if model_name == "Hybrid":

    bm25_weight = st.slider(
        "BM25 Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    embedding_weight = (
        1.0 - bm25_weight
    )

    st.info(
        f"Embedding Weight: {embedding_weight:.2f}"
    )

top_k = st.slider(
    "Top K Results",
    min_value=1,
    max_value=50,
    value=10
)

use_query_refinement = st.checkbox(
    "Enable Query Refinement",
    value=False
)

use_clustering = False

if model_name in [
    "TF-IDF",
    "Embedding"
]:

    use_clustering = st.checkbox(
        "Enable Clustering Search"
    )

run_evaluation = st.button(
    "Run Evaluation"
)

if run_evaluation:

    queries, qrel_dict = load_queries_qrels(
        dataset,
        doc_ids
    )

    experiment_context = {

        "queries": queries,
        "qrel_dict": qrel_dict,

        "tfidf_vectorizer":
            indexes["tfidf_vectorizer"],

        "tfidf_matrix":
            indexes["tfidf_matrix"],

        "bm25":
            indexes["bm25"],

        "model":
            indexes["model"],

        "document_embeddings":
            indexes["document_embeddings"],

        "doc_ids":
            doc_ids,

        "preprocess_text":
            preprocess_text,

        "get_tfidf_scores":
            get_tfidf_scores,

        "get_bm25_scores":
            get_bm25_scores,

        "get_embedding_scores":
            get_embedding_scores,

        "normalize":
            normalize,

        "refine_query":
            refine_query,

        "use_query_refinement":
            use_query_refinement
    }

    results = run_experiment(
        experiment_context
    )

    results_df = build_results_dataframe(
        results
    )

    st.subheader(
        "Evaluation Results"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.subheader(
        "MAP Comparison"
    )

    st.bar_chart(
        results_df.set_index(
            "Model"
        )["MAP"]
    )

    st.subheader(
        "nDCG@10 Comparison"
    )

    st.bar_chart(
        results_df.set_index(
            "Model"
        )["nDCG@10"]
    )

    best_model = results_df.loc[
        results_df["MAP"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(MAP={best_model['MAP']})"
    )

query = st.text_input(
    "Enter Query"
)

search_clicked = st.button(
    "Search"
)


if search_clicked:

    if query.strip() == "":

        st.warning(
            "Please enter a query"
        )

    else:
        
        cluster_id, topic_name = detect_topic(
            query,
            indexes["model"],
            indexes["cluster_model"],
            indexes["topic_names"]
        )

        cluster_labels = indexes["cluster_labels"]

        cluster_size = (
            cluster_labels == cluster_id
        ).sum()

        st.success(
            f"Predicted Topic: "
            f"{topic_name}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Topic ID",
                cluster_id
            )

        with col2:
            st.metric(
                "Documents In Topic",
                cluster_size
            )

        results = search_documents(
            query=query,
            model_name=model_name,
            indexes=indexes,
            doc_ids=doc_ids,
            preprocess_text=preprocess_text,
            top_k=top_k,
            use_query_refinement=use_query_refinement,
            use_clustering=use_clustering,

            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
        )

        st.write(
            f"Query Used: {query}"
        )

        st.subheader(
            "Results"
        )


        for rank, doc_id in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Result #{rank} - {doc_id}"
            ):

                document = get_document(
                    dataset_name,
                    doc_id
                )

                if document:

                    st.write(
                        document
                    )

                else:

                    st.error(
                        "Document Not Found"
                    )

            st.divider()

            