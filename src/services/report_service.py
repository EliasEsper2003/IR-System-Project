import numpy as np
import pandas as pd


def build_results_dataframe(results):

    return pd.DataFrame([

        {
            "Model": "TF-IDF",
            "Recall@10": round(np.mean(results["tfidf_recalls"]), 4),
            "Precision@10": round(np.mean(results["tfidf_precisions"]), 4),
            "MAP": round(np.mean(results["tfidf_maps"]), 4),
            "nDCG@10": round(np.mean(results["tfidf_ndcgs"]), 4),
        },

        {
            "Model": "BM25",
            "Recall@10": round(np.mean(results["bm25_recalls"]), 4),
            "Precision@10": round(np.mean(results["bm25_precisions"]), 4),
            "MAP": round(np.mean(results["bm25_maps"]), 4),
            "nDCG@10": round(np.mean(results["bm25_ndcgs"]), 4),
        },

        {
            "Model": "Embedding",
            "Recall@10": round(np.mean(results["embedding_recalls"]), 4),
            "Precision@10": round(np.mean(results["embedding_precisions"]), 4),
            "MAP": round(np.mean(results["embedding_maps"]), 4),
            "nDCG@10": round(np.mean(results["embedding_ndcgs"]), 4),
        },

        {
            "Model": "Hybrid",
            "Recall@10": round(np.mean(results["hybrid_recalls"]), 4),
            "Precision@10": round(np.mean(results["hybrid_precisions"]), 4),
            "MAP": round(np.mean(results["hybrid_maps"]), 4),
            "nDCG@10": round(np.mean(results["hybrid_ndcgs"]), 4),
        },

        {
            "Model": "Hybrid Serial",
            "Recall@10": round(np.mean(results["serial_recalls"]), 4),
            "Precision@10": round(np.mean(results["serial_precisions"]), 4),
            "MAP": round(np.mean(results["serial_maps"]), 4),
            "nDCG@10": round(np.mean(results["serial_ndcgs"]), 4),
        }

    ])