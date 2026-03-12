# Quora Question Pair Duplicate Detection

## Project Overview

This project aims to detect whether two questions asked on Quora are **duplicates** or not. Many users ask the same question in different ways, which creates redundancy in the platform. Detecting duplicate questions helps improve search quality and reduce repeated content.

In this project, **Natural Language Processing (NLP)** techniques and **Machine Learning models** are used to analyze question pairs and determine whether they are duplicates.

---

## Dataset

The dataset used in this project is the **Quora Question Pair Dataset**.

### Dataset Features

| Column | Description |
|------|-------------|
| id | Unique identifier for each question pair |
| qid1 | Unique ID of question 1 |
| qid2 | Unique ID of question 2 |
| question1 | First question |
| question2 | Second question |
| is_duplicate | Target variable (1 = duplicate, 0 = not duplicate) |

For faster computation, a **sample of 30,000 rows** was used from the original dataset.

---

## Data Preprocessing

Several preprocessing techniques were applied to clean and normalize the text data.

### Preprocessing Steps

- Convert text to lowercase
- Remove HTML tags
- Remove punctuation
- Replace special characters (`%`, `$`, `@`)
- Expand English contractions
- Normalize large numbers (e.g., 1000000 → 1m)
- Remove unwanted characters using regex

---

## Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify patterns between duplicate and non-duplicate questions.

### Key Observations

- Duplicate questions often have **higher word overlap**
- Similar **question lengths**
- Higher **text similarity scores**

### Visualizations Used

- Distribution of question lengths
- Word count distributions
- Common word overlap distribution
- Word share distribution
- Pairplots for feature relationships

Libraries used for visualization:

```python
seaborn
matplotlib
```

---

## Feature Engineering

Multiple features were created to capture the similarity between question pairs.

### Basic Features

- `q1_len` – Length of question 1
- `q2_len` – Length of question 2
- `q1_num_words` – Number of words in question 1
- `q2_num_words` – Number of words in question 2

---

### Word Overlap Features

- `word_common` – Number of common words
- `word_total` – Total number of unique words
- `word_share` – Ratio of common words

---

### Token-Based Features

- `cwc_min`
- `cwc_max`
- `csc_min`
- `csc_max`
- `ctc_min`
- `ctc_max`
- `first_word_eq`
- `last_word_eq`

These features measure the similarity between tokens and stopwords of the two questions.

---

### Length-Based Features

- `abs_len_diff` – Absolute difference in token lengths
- `mean_len` – Average token length

---

### Fuzzy Matching Features

Using **FuzzyWuzzy**, several string similarity metrics were computed:

- `fuzz_ratio`
- `fuzz_partial_ratio`
- `token_sort_ratio`
- `token_set_ratio`

These features help capture **semantic similarity even when wording differs**.

---

## Text Vectorization

Text data was converted into numerical form using **Bag of Words (BoW)** with `CountVectorizer`.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=3000)
```

Both questions were vectorized separately and combined to create **6000 text features**.

---

## Final Dataset

The final dataset consists of:

- Engineered similarity features
- Fuzzy matching features
- Bag-of-Words features

Final dataset shape:

```
30000 rows × ~6000 features
```

---

## Machine Learning Models

Three machine learning models were trained and evaluated.

### Random Forest

Random Forest is an ensemble learning method based on multiple decision trees.

**Accuracy:**  
```
78.25%
```

---

### XGBoost

XGBoost is a powerful gradient boosting algorithm widely used in machine learning competitions.

**Accuracy:**  
```
79.33%
```

Best performing model in this project.

---

### Support Vector Machine (SVM)

A linear SVM classifier was also tested.

**Accuracy:**  
```
74.65%
```

---

## Model Evaluation

The models were evaluated using:

- Accuracy Score
- Confusion Matrix

Example confusion matrix:

```
[[3071  741]
 [ 780 1408]]
```

---

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- NLTK
- FuzzyWuzzy
- BeautifulSoup
- Regex

---

## Project Workflow

```
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Text Vectorization
      ↓
Model Training
      ↓
Model Evaluation
```

---

## Results

The **XGBoost model achieved the best performance with ~79% accuracy**, showing that combining engineered features with text vectorization can effectively detect duplicate questions.

---

## Future Improvements

Possible improvements include:

- Using **TF-IDF instead of Bag of Words**
- Implementing **Word Embeddings (Word2Vec / GloVe)**
- Using **Deep Learning models (LSTM / BERT)**
- Hyperparameter tuning
- Training on the **full dataset instead of sampling**

---

## Author

Tulsi Rajora
