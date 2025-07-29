# Feature Engineering and Text Vectorization Techniques

This repository demonstrates key techniques in feature engineering—specifically polynomial feature generation, interaction features, and converting text data into numerical vectors using TF-IDF. These steps are essential for enhancing model inputs and improving predictive performance.

---

## 📌 Contents

### 1. 🧮 Feature Generation (Polynomial Features)
- **Objective**: Capture non-linear relationships by creating new features as polynomial combinations of existing features.
- **Method Used**: `PolynomialFeatures` from `sklearn.preprocessing`
- **Example**: For features `Bedrooms`, `Bathrooms`, and `Square Footage`, it generates:
  - Squared terms (e.g., `Bedrooms^2`)
  - Interaction terms (e.g., `Bedrooms * Bathrooms`)
- **Usage**: Can improve performance of linear models on non-linear data.

---

### 2. 🛠 Feature Engineering Concepts

#### ➤ Feature Creation (Interaction Features)
- **What It Means**: Combining two or more existing features to form new ones that may better capture relationships in data.
- **Why It Matters**: Some models (like linear regression) cannot capture interaction effects unless we explicitly create them.
- **Example**: `Bedrooms * Bathrooms` may be a better predictor of home price than either alone.

#### ➤ Feature Extraction
- **What It Means**: Converting raw, unstructured data into structured numerical formats.
- **Context Shown**: Extracting numeric values from text using TF-IDF.

---

### 3. 🔤 Text Data Vectorization (TF-IDF)

- **Objective**: Convert text documents into numerical vectors that machine learning models can understand.
- **Method Used**: `TfidfVectorizer` from `sklearn.feature_extraction.text`
- **How It Works**:
  - **TF (Term Frequency)**: How often a word appears in a document.
  - **IDF (Inverse Document Frequency)**: Reduces weight of commonly occurring words across all documents.
- **Result**: Sparse matrix of weights indicating word importance per document.
- **Use Case**: Essential in Natural Language Processing tasks like classification, clustering, and search ranking.

---

## 🧾 Sample Output (TF-IDF Interpretation)

Each row represents a document. Each column represents a word with its TF-IDF weight:

| learning | machine | natural | processing | ... |
|----------|---------|---------|------------|-----|
| 0.60     | 0.46    | 0.00    | 0.00       | ... |
| 0.00     | 0.34    | 0.00    | 0.00       | ... |
| 0.00     | 0.00    | 0.39    | 0.39       | ... |

---

## 🛠 Technologies Used
- Python 3.x
- pandas
- scikit-learn

---

## 💡 Use Cases
- Improve linear model expressiveness through polynomial and interaction terms.
- Convert raw text to meaningful numerical features for text-based ML tasks.
- Understand key differences between feature creation and extraction.

---

## 📬 Contact
Feel free to fork the repo, contribute, or raise issues for enhancements.
