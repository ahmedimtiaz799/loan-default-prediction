Loan Default Prediction 🏦📉
This project focuses on predicting whether a customer will default on a personal loan using supervised machine learning techniques. The dataset is moderately imbalanced, so class balancing, correlation analysis, feature engineering, and model tuning are applied to improve predictive performance.

📂 Dataset
Source: Kaggle – Loan Default Prediction Dataset

Description: The dataset contains financial and demographic information for 1,000 individuals. The goal is to predict whether a customer will accept a personal loan (1) or not (0). The dataset is slightly imbalanced, with only ~9.6% of customers accepting the loan.

⚙️ Technologies Used
Python

pandas, NumPy

seaborn, matplotlib

scikit-learn

XGBoost

Jupyter Notebook

🔍 Project Workflow
1. Data Loading & Initial Exploration
Loaded dataset using pandas.read_csv()

Used .head(), .info(), .shape, and .describe() to inspect the data structure

Dropped irrelevant columns: 'ID', 'ZIP Code'

2. Data Cleaning
Fixed negative values in the Experience column by replacing them with NaN

Filled missing values in Experience with the median

Confirmed no remaining missing or duplicate values

3. Exploratory Data Analysis (EDA)
Target Variable (Personal Loan): Checked class balance using sns.countplot() and value_counts()

Numerical Feature Distributions:

Income

CCAvg (Credit Card Avg.)

Experience

Age

Mortgage

Categorical Feature Distributions:

Family size

Education level

Securities Account

CD Account

Online

Credit Card ownership

4. Boxplot Analysis
Plotted boxplots to compare numerical features (Income, CCAvg, Age, Experience, Mortgage) against the loan approval status to observe differences in distributions.

5. Correlation Analysis
Computed and visualized feature correlations with the target Personal Loan

Found strong positive correlations with:

CD Account

Income

CCAvg

6. Model Training
🎯 Features (X):
All columns except Personal Loan (target) and dropped identifiers

🏷️ Target (y):
Personal Loan

📌 Train/Test Split:
Used train_test_split with stratification to maintain class ratio.

🔸 Decision Tree Classifier
class_weight='balanced', max_depth=5

Performance:

Precision (1): 0.66

Recall (1): 0.99

F1-Score (1): 0.79

Accuracy: 0.95

🔸 XGBoost Classifier (with Tuning)
Applied hyperparameter tuning to improve generalization

Performance:

Precision (1): 0.85

Recall (1): 0.98

F1-Score (1): 0.91

Accuracy: 0.98

📈 Evaluation Metrics
Classification Report (Precision, Recall, F1-Score)

Confusion Matrix

Accuracy

Macro and Weighted Averages

📊 Visualizations
Loan distribution: sns.countplot

Histograms for Income, Age, Experience, CCAvg, and Mortgage (by Loan class)

Count plots for categorical features (Education, Family, etc.)

Boxplots to compare numeric features against loan approval

Correlation barplot showing the relationship of each feature with the target

✅ Results Summary
Model	F1-Score (Loan=1)	Precision (Loan=1)	Recall (Loan=1)	Accuracy
Decision Tree	0.79	0.66	0.99	0.95
XGBoost (Tuned)	0.91	0.85	0.98	0.98
