# Insurance Claim Fraud Detection

## Overview

This project develops and evaluates machine learning models to detect fraudulent insurance claims by analyzing patterns in policyholder information, policy details, incident characteristics, and financial claim data. The goal is to build a predictive system that assigns risk scores to claims, enabling insurance companies to prioritize investigations and reduce financial losses from fraud.

## Business Context

Insurance fraud costs the U.S. insurance industry an estimated **$308.6 billion annually**. A predictive model that reliably flags suspicious claims enables insurers to:
- Prioritize investigative resources efficiently
- Reduce financial losses from undetected fraud
- Accelerate processing for legitimate claims
- Lower premiums for honest policyholders

The techniques demonstrated here are transferable across fraud detection domains including healthcare, credit card, and financial services.

## Dataset

The analysis uses the **Insurance Claim Fraud Detection** dataset from Kaggle:

- [Kaggle Dataset](https://www.kaggle.com/code/niteshyadav3103/insurance-fraud-detection-using-12-models/data?select=insurance_claims.csv)
- **Records**: 1,000 insurance claim records
- **Features**: 40 features including demographics, policy details, incident information, and financial data
- **Target Variable**: `fraud_reported` (Y/N)
  - `Y` = Fraudulent claim (24.7% - 247 claims)
  - `N` = Legitimate claim (75.3% - 753 claims)

### Feature Categories

**Customer Demographics:**
- Age, sex, education level, occupation, hobbies, relationship status

**Policy Details:**
- Policy state, CSL, deductible, annual premium, umbrella limit, months as customer

**Incident Information:**
- Incident type, collision type, severity, location, authorities contacted, witnesses, police report availability

**Financial Details:**
- Capital gains/losses, total claim amount, injury/property/vehicle claim breakdowns

**Vehicle Information:**
- Auto make, model, year

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of fraudulent vs. legitimate claims
- Feature correlation analysis
- Outlier detection using box plots
- Identification of strong predictive features

### 2. Data Preprocessing
- **Missing value handling**: Replaced NA values with mode values
- **Feature engineering**: Converted categorical target to binary numeric
- **Feature dropping**: Removed non-predictive columns (policy_number, dates, incident_location)
- **Train-test split**: 80/20 split with stratification to maintain fraud ratio
- **Feature encoding**: 
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- **Class imbalance handling**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to training set only

### 3. Model Training & Hyperparameter Tuning
- **Grid Search Cross-Validation** (5-fold) for optimal hyperparameters
- **Scoring metric**: F1 score (balances precision and recall)
- **Training approach**: Models trained on SMOTE-balanced data, evaluated on original imbalanced test set

### 4. Model Evaluation
Comprehensive evaluation using multiple metrics:
- **Accuracy**: Overall correctness
- **Precision**: Fraud prediction reliability (minimize false positives)
- **Recall**: Fraud detection rate (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Ability to distinguish between classes
- **PR AUC**: Performance on imbalanced data
- **Confusion Matrices**: Detailed prediction breakdown
- **Feature Importance Analysis**: Understanding key fraud indicators

## Models Evaluated

Five classification algorithms with extensive hyperparameter tuning:

### 1. Logistic Regression
- Hyperparameters: C (regularization), penalty (l1/l2), solver
- Provides interpretable coefficients for feature importance

### 2. K-Nearest Neighbors (KNN)
- Hyperparameters: n_neighbors, weights (uniform/distance), metric
- Instance-based learning approach

### 3. Random Forest
- Hyperparameters: n_estimators, max_depth, min_samples_leaf, min_samples_split
- Ensemble of decision trees with feature importance

### 4. Gradient Boosting
- Hyperparameters: n_estimators, learning_rate, max_depth, subsample
- Sequential ensemble method with strong predictive power

### 5. XGBoost
- Hyperparameters: n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree
- Optimized gradient boosting with regularization

## Key Findings

### Data Insights

1. **Moderate Class Imbalance**  
   3:1 ratio (75% legitimate, 25% fraudulent)

2. **Strong Signal Features**  
   - **Incident severity**: Highest mutual information with fraud; Major Damage claims show distinctly higher fraud rates
   - **Claim amount breakdowns**: total_claim_amount, injury_claim, property_claim, vehicle_claim provide rich continuous signals
   - **Incident type**: Single Vehicle Collision, Vehicle Theft show clear fraud rate variation
   - **Police report availability**: Missing reports correlate with higher fraud likelihood

3. **Data Quality Issues**  
   - Missing values in collision_type, property_damage, police_report_available
   - 91 null values in authorities_contacted (9.1%)
   - Successfully handled through mode imputation

4. **Feature Correlations**  
   - High correlation between injury_claim, property_claim, vehicle_claim and total_claim_amount (they sum to total)
   - Strong correlation between age and months_as_customer

### Model Performance

All models achieved strong performance, with **Random Forest** and **XGBoost** demonstrating the best overall results:

**Top Performers:**
- **Random Forest** & **XGBoost**: F1 scores ~0.95-0.98, excellent balance of precision and recall
- **Gradient Boosting**: Strong performer across all metrics
- **Logistic Regression**: Solid baseline with interpretable results
- **KNN**: Competitive performance, slightly lower precision

**Key Metrics:**
- **Accuracy**: >90% across all models, with top performers reaching 95-97%
- **Precision**: >0.90 for best models (minimizes false fraud alerts)
- **Recall**: >0.90 for best models (catches most fraudulent claims)
- **F1 Score**: 0.95+ for Random Forest and XGBoost
- **ROC AUC & PR AUC**: Consistently >0.95 for ensemble models

### Feature Importance

The most important features are:
1. **Incident-related features**: severity, type, collision type
2. **Financial features**: total_claim_amount, claim breakdowns
3. **Policy and demographic features**: moderate contribution
4. **Categorical encodings**: One-hot encoded features

## Visualizations

The notebook includes comprehensive visualizations:
- Distribution plots of numerical features by fraud status
- Count plots for categorical features vs. fraud
- Box plots for outlier detection
- Correlation heatmap
- Confusion matrices (5 models side-by-side)
- ROC curves comparison
- Precision-Recall curves comparison
- Feature importance plots (4 models)
- Model performance comparison across 6 metrics

## Deployment Recommendations

### Model Selection
Deploy **Random Forest** or **XGBoost** as the primary fraud detection model based on:
- Highest F1 scores and balanced precision-recall
- Strong feature importance interpretability
- Robust performance across all evaluation metrics

## Notebook

- [View the Analysis Notebook](./insurance_claims_fraud_detection.ipynb)

## Repository Structure

```text
capstone_final/
├── insurance_claims_fraud_detection.ipynb
├── README.md
└── data/
    └── insurance_claims.csv
```

## Requirements

### Python Libraries
```
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
imbalanced-learn
xgboost
```

### Installation
```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn imbalanced-learn xgboost
```

## How to Run

1. **Clone or download** this repository
2. **Ensure the dataset** is available at `data/insurance_claims.csv`
3. **Install dependencies** listed above
4. **Open the notebook** in Jupyter or VS Code:
   ```bash
   jupyter notebook insurance_claims_fraud_detection.ipynb
   ```
   Or open directly in VS Code with Jupyter extension
5. **Run cells sequentially** to reproduce the analysis

## Future Enhancements

1. **Larger datasets:**:
   - Validate the approach on a larger dataset to confirm these patterns hold up at scale.
2. **Threshold tuning:** 
   - Adjust the classification threshold beyond the default 0.50 to optimize the precision-recall tradeoff for specific business requirements.
3. **Model Improvements**:
   - Deep learning approaches (neural networks, autoencoders)
   - Ensemble stacking combining multiple model predictions

## Author

Berkeley Professional Certificate in Machine Learning and Artificial Intelligence - Capstone Project

## License

This project is for educational purposes.
3. Install dependencies if needed.
4. Run the notebook cells in order.

Example install command:

```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn imbalanced-learn
```
