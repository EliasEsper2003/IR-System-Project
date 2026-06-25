import ir_datasets

import os

from services.storage_service import (
    save_pickle,
    load_pickle
)


def load_dataset_data(
    dataset_name,
    preprocess_text
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

        raw_docs_path = (
            f"{save_dir}/raw_documents.pkl"
        )

        processed_docs_path = (
            f"{save_dir}/processed_documents.pkl"
        )

        doc_ids_path = (
            f"{save_dir}/doc_ids.pkl"
        )

        if (
            os.path.exists(raw_docs_path)
            and os.path.exists(processed_docs_path)
            and os.path.exists(doc_ids_path)
        ):

            print("Loading Saved Documents...")

            raw_documents = load_pickle(
                raw_docs_path
            )

            processed_documents = load_pickle(
                processed_docs_path
            )   

            doc_ids = load_pickle(
                doc_ids_path
            )

            dataset = ir_datasets.load(
                dataset_name
            )

            print(
                "Loaded Documents:",
                len(doc_ids)
            )

            return (
                dataset,
                raw_documents,
                processed_documents,
                doc_ids
            )        

        dataset = ir_datasets.load(dataset_name)

        print("Dataset:", dataset_name)

        processed_documents = []
        raw_documents = []
        doc_ids = []

        for i, doc in enumerate(dataset.docs_iter()):


            raw_documents.append(doc.text)

            processed_documents.append(
                preprocess_text(doc.text)
            )

            doc_ids.append(doc.doc_id)

        print("Loaded Documents:", len(doc_ids))

        save_pickle(
            raw_documents,
            raw_docs_path
        )

        save_pickle(
            processed_documents,
            processed_docs_path
        )

        save_pickle(
            doc_ids,
            doc_ids_path
        )

        print("Documents Saved")

        return (
            dataset,
            raw_documents,
            processed_documents,
            doc_ids
        )


def load_queries_qrels(
    dataset,
    doc_ids
):

    queries = list(
        dataset.queries_iter()
    )

    qrels = list(
        dataset.qrels_iter()
    )

    available_docs = set(doc_ids)

    qrel_dict = {}

    for qrel in qrels:

        if qrel.doc_id not in available_docs:
            continue

        if qrel.query_id not in qrel_dict:
            qrel_dict[qrel.query_id] = set()

        qrel_dict[qrel.query_id].add(
            qrel.doc_id
        )

    filtered_queries = []

    for q in queries:

        if q.query_id in qrel_dict:
            filtered_queries.append(q)

    queries = filtered_queries

    print("Queries:", len(queries))
    print("Qrels:", len(qrel_dict))

    return queries, qrel_dict