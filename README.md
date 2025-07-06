### ✅ **Task 1: Exploratory Data Analysis (EDA) and Data Preprocessing**

In this task, we performed initial data exploration and preprocessing to prepare the customer complaints dataset for downstream NLP and machine learning tasks. The goal was to clean and structure the data to improve the quality of insights derived from unstructured complaint narratives.

#### 🔍 Exploratory Data Analysis

* Analyzed the distribution of complaints across different products.
* Calculated and visualized the word count of complaint narratives to identify very short and very long entries.
* Identified the number of complaints with and without textual narratives.

#### 🧹 Data Filtering & Cleaning

* Filtered the dataset to include only records related to five key product categories:

  * Credit card
  * Personal loan
  * Buy Now, Pay Later (BNPL)
  * Savings account
  * Money transfers
* Removed records with empty or null `Consumer complaint narrative` fields.
* Cleaned the narrative text to improve embedding quality:

  * Lowercased all text.
  * Removed special characters and excessive whitespace.
  * Removed boilerplate phrases frequently found in complaint templates (e.g., “I am writing to file a complaint”, “To whom it may concern”).

#### 💾 Output

* Saved the cleaned and filtered dataset to:
  `data/filtered_complaints.csv`
  This file will be used for subsequent tasks including embedding generation, topic modeling, and dashboard development.

---

### ✅ **Task 2: Text Chunking, Embedding, and Vector Store Indexing**

This task focused on transforming cleaned complaint narratives into a structured vector format optimized for semantic search and retrieval.

#### 🧱 Text Chunking

* Implemented text chunking using **LangChain’s `RecursiveCharacterTextSplitter`** to split long narratives into smaller, overlapping chunks.
* Experimented with various `chunk_size` and `chunk_overlap` settings to balance semantic coherence and chunk coverage.
* Chosen configuration: `chunk_size=500`, `chunk_overlap=50`.

#### 🔍 Embedding Generation

* Selected and used the **`all-MiniLM-L6-v2`** model from `sentence-transformers` for its efficiency and strong performance on semantic tasks.
* Generated a 384-dimensional embedding for each text chunk.

#### 📦 Vector Indexing

* Used **FAISS** to build a vector store for fast and scalable similarity search.
* Stored metadata for each chunk (e.g., `Complaint ID`, `Product`, `Issue`) in a separate file to maintain traceability.

#### 💾 Output Files

* `data/complaints_index.faiss` – FAISS index of chunk embeddings

These outputs will power semantic search, clustering, and retrieval-augmented analysis in subsequent tasks.


