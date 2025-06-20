# 🏦 Loan Default Prediction using Machine Learning 📉

This project aims to predict whether a customer will **default on a personal loan** using **supervised machine learning techniques**. The dataset is moderately imbalanced, so techniques like **class balancing**, **EDA**, **feature engineering**, and **model tuning** are applied to enhance performance.

---

## 📂 Dataset

* **Source**: [Kaggle – Loan Default Prediction Dataset](https://www.kaggle.com/)
* **Description**:
  The dataset includes **financial and demographic information** for **1,000 individuals**.
  The goal is to predict whether a customer will **accept a personal loan**:

  * `0` → No
  * `1` → Yes

  \~9.6% of customers accepted a loan, indicating moderate class imbalance.

---

## ⚙️ Technologies Used

* Python
* **Libraries**: `pandas`, `NumPy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
* Jupyter Notebook

---

## 🔍 Project Workflow

### 🧾 1. Data Loading & Initial Exploration

* Loaded data with `pandas.read_csv()`
* Inspected shape, column names, data types, and summary statistics
* Dropped irrelevant columns: `ID`, `ZIP Code`

### 🧹 2. Data Cleaning

* Replaced **negative values** in `Experience` with `NaN`
* Imputed missing `Experience` values using **median**
* Checked for duplicates and confirmed data integrity

### 📊 3. Exploratory Data Analysis (EDA)

#### 🎯 Target Distribution

* Visualized class imbalance using `sns.countplot()` and `value_counts()`

#### 📈 Numerical Features

* Plotted histograms for:

  * `Income`
  * `CCAvg` (Credit Card Avg.)
  * `Experience`
  * `Age`
  * `Mortgage`

#### 🧮 Categorical Features

* Visualized count plots for:

  * `Family` (size)
  * `Education`
  * `Securities Account`
  * `CD Account`
  * `Online`
  * `CreditCard`

#### 📦 Boxplots

* Compared distributions of numeric features (`Income`, `CCAvg`, etc.) against loan approval status

#### 🔗 Correlation Analysis

* Generated correlation heatmap
* Found strong positive correlation with:

  * `CD Account`
  * `Income`
  * `CCAvg`

---

## 🤖 Model Training & Evaluation

### 🧠 Features & Target

* `X`: All features excluding the `Personal Loan` column and identifiers
* `y`: `Personal Loan` column (target)

### 🔀 Train/Test Split

* Used `train_test_split` with `stratify=y` to maintain class distribution

---

### 🌳 Decision Tree Classifier

* Parameters: `class_weight='balanced'`, `max_depth=5`

| Metric    | Loan = 1 (Positive Class) |
| --------- | ------------------------- |
| Precision | 0.66                      |
| Recall    | 0.99                      |
| F1-Score  | 0.79                      |
| Accuracy  | 0.95                      |

---

### ⚡ XGBoost Classifier (with Tuning)

* Applied grid search and tuning for better generalization

| Metric    | Loan = 1 (Positive Class) |
| --------- | ------------------------- |
| Precision | 0.85                      |
| Recall    | 0.98                      |
| F1-Score  | 0.91                      |
| Accuracy  | 0.98                      |

---

## 📈 Evaluation Metrics

* **Classification Report** (Precision, Recall, F1-Score)
* **Confusion Matrix**
* **Overall Accuracy**
* **Macro & Weighted Averages**

---

## 📊 Key Visualizations

* `sns.countplot` for class distribution
* Histograms for numeric features by loan class
* Count plots for categorical features
* Boxplots comparing features with loan status
* Correlation heatmap / barplot

---

## ✅ Results Summary

| Model           | F1-Score | Precision | Recall | Accuracy |
| --------------- | -------- | --------- | ------ | -------- |
| Decision Tree   | 0.79     | 0.66      | 0.99   | 0.95     |
| XGBoost (Tuned) | 0.91     | 0.85      | 0.98   | 0.98     |

---

## 📌 Conclusion

* **XGBoost** significantly outperforms Decision Tree on this moderately imbalanced dataset.
* Strong predictors include **Income**, **CD Account**, and **Credit Card Spending (CCAvg)**.
* The pipeline includes **cleaning, visualization, feature analysis, and hyperparameter tuning**, making the model reliable for loan risk assessment.






