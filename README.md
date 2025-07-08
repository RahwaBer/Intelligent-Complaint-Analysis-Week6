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

---

### ✅ **Task 3: Building the RAG Core Logic and Evaluation**

This task focused on implementing the core components of a **Retrieval-Augmented Generation (RAG)** system to allow internal stakeholders to query customer complaint data and receive relevant, evidence-based answers.

#### 🔍 Retriever Implementation

* Created a function that:

  * Embeds user questions using the same model as the complaint chunks (`all-MiniLM-L6-v2`)
  * Performs a similarity search using a FAISS index
  * Retrieves the top-`k` most relevant complaint chunks (`k=5`)

#### 💬 Generator Implementation

* Designed a robust prompt template to guide the LLM:

  * Instructs the model to act as a financial analyst assistant
  * Encourages strict adherence to provided context
  * Gracefully handles insufficient context
* Combined the user’s question, retrieved chunks, and prompt into a complete input
* Used a locally hosted or Hugging Face model (e.g., Mistral) to generate grounded answers

#### 📊 Qualitative Evaluation

* Created a list of 10 representative questions based on real user needs
* Ran the full RAG pipeline for each question
* Collected results in a structured Markdown evaluation table including: Question
  
#### 🧾 Outcome

* A working RAG pipeline capable of answering product team questions with traceable complaint excerpts
* Evaluation results to guide prompt tuning, retriever quality, and future improvements

