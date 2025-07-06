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


