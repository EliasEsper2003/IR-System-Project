# 🔍 Advanced Enterprise Information Retrieval (IR) System

A scalable, distributed, and high-performance custom Search Engine built in **Python**, designed according to **Service-Oriented Architecture (SOA)** and **Clean Architecture** principles. Tested and evaluated across two large-scale standard IR datasets containing **>200,000 documents**.

---

## 📂 Datasets Used

* **TREC-COVID:** Evaluated for medical and scientific natural language queries.
* **Touché 2020:** Evaluated for argumentative and conversational search scenarios.

---

## 🌟 Architecture & Core Features

### 🏗️ Service-Oriented Architecture (SOA)
The system is decoupled into independent microservices communicating via robust protocols:
* **API Gateway & Streamlit UI:** System entry point providing a user-friendly interface for queries, parameter tuning, and hybrid execution selection.
* **Data Preprocessing Service:** Handles text normalization, lemmatization, stemming, and stop-word removal.
* **Indexing Service:** Manages fast inverted indices optimized for high-throughput querying.
* **Retrieval Service:** Executes search algorithms across multiple document representations.
* **Query Refinement Service:** Implements query expansion, spell correction, and search history weighting.
* **Ranking & Evaluation Service:** Calculates relevance scores and measures precision and ranking performance metrics.

---

### 🧬 Representation Models & Search Paradigms
* **Vector Space Model (VSM):** TF-IDF representation with Cosine Similarity ranking.
* **Dense Embeddings:** Semantic representation utilizing **Word2Vec** and **BERT** models.
* **Probabilistic Model (BM25):** Okapi BM25 implementation with dynamic parameter tuning from the UI.
* **Hybrid Representation Engine:**
  * **Parallel Hybrid:** Executes multiple models concurrently and fuses scores using advanced **Score Fusion Methods**.
  * **Serial Hybrid:** Cascades representations sequentially to refine candidate document pools.

---

### 🚀 Advanced Features
* **Documents Clustering:** Unsupervised grouping of search results to categorize retrieved information dynamically.
* **Topic Detection:** Automatically identifies and extracts latent topics from document clusters.

---

## 📊 Evaluation Metrics

Evaluated thoroughly using qrel relevance judgements:
* **MAP (Mean Average Precision)**
* **nDCG (Normalized Discounted Cumulative Gain)**
* **Precision@10** & **Recall**

---

## ⚙️ Quick Start & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/EliasEsper2003/IR-System-Project.git](https://github.com/EliasEsper2003/IR-System-Project.git)
   cd IR-System-Project
