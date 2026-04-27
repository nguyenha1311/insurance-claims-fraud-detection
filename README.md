# 🚗 Vehicle Claim Fraud Detection

## 📋 Project Overview
This project investigates vehicle insurance claim fraud using machine learning classification techniques. Insurance companies lose significant revenue to fraudulent claims each year. The goal is to build a predictive model that assigns a fraud risk score to each claim, helping investigators prioritize cases for review.

**Dataset:** [Kaggle — Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data?select=fraud_oracle.csv)

---

## 📓 Jupyter Notebook
👉 [View the full analysis notebook here](./prompt.ipynb)

---

## 🔍 Summary of Findings

### 1. Class Imbalance
The dataset is heavily imbalanced — approximately **94% legitimate** claims and only **6% fraudulent**. SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set to address this before modeling.

### 2. Data Quality
- A significant number of records had **Age = 0**, which is a data entry error. These were imputed with the **median age** of valid records.
- No missing values or duplicate rows were found after cleaning.

### 3. Key Fraud Indicators (EDA)
| Feature | Finding |
|---------|---------|
| **Fault** | Policy Holder fault claims show a higher fraud rate than Third Party fault |
| **Policy Type** | All Perils and Collision policies have higher fraud density |
| **Police Report** | Absence of a police report strongly correlates with fraud |
| **Numerical features** | Age, Deductible, DriverRating show weak linear correlation with fraud — suggesting non-linear patterns |

### 4. Baseline Model — Logistic Regression
A Logistic Regression model was trained as the baseline classifier.

| Metric | Result |
|--------|--------|
| **Accuracy** | High — but misleading due to class imbalance |
| **Recall** | Reasonable — model catches a meaningful portion of fraud cases |
| **Precision** | Lower — some false positives present (acceptable at baseline) |
| **ROC AUC** | Significantly above 0.5 — model has meaningful ability to distinguish fraud from legitimate claims |
| **F1 Score** | Reflects the Precision-Recall trade-off; optimal threshold is below the default 0.5 |

### 5. Limitations & Next Steps
- Logistic Regression assumes **linear decision boundaries**, which may not capture the complex patterns in this dataset.
- Future work will evaluate **KNN** and **Random Forest** classifiers to model non-linear interactions.
- **Threshold tuning** will be explored to improve Recall at the cost of acceptable Precision loss.

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn (SMOTE)
- Seaborn, Matplotlib

---

## 📁 Project Structure
```
capstone_submission_1_EDA/
│
├── data/
│   └── fraud_oracle.csv
├── prompt.ipynb        # Main analysis notebook
└── README.md
```
