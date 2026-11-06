# Predicting Student Dropout in Higher Education Using Supervised Machine Learning Classification

**Developers:** Winnie Wasike, Naima Bashir, Mark Lemute
**Course:** Data Mining

## Overview
Student dropout costs students, universities and the economy. This project builds a supervised **binary classification** model that predicts whether a student will **drop out or not**, using demographic, socio-economic and academic data, so institutions can identify at-risk students early.

## Dataset
- **Source:** [UCI Machine Learning Repository: Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (Realinho, Machado, Baptista & Martins, 2021)
- **Size:** 4,424 students, 34 features + 1 target
- **Target:** the original three classes were merged into two: **Dropout** (1,421) and **Non-Dropout** (3,003 = Enrolled + Graduate)
- **Quality:** no missing values, no duplicates
- `data.csv` is included so the project runs without downloading anything.

## Methodology
1. Load and clean the data (check missing values and duplicates)
2. Merge Enrolled and Graduate into Non-Dropout (binary target)
3. Stratified 80/20 train-test split (3,539 train / 885 test)
4. Standard scaling of features
5. SMOTE oversampling, applied inside the training folds only (no data leakage)
6. Train Decision Tree, Random Forest, Logistic Regression and XGBoost
7. Validate with 10-fold stratified cross-validation and the held-out test set
8. Evaluate with accuracy, precision, recall, F1-score and a confusion matrix

## Results
| Model | CV Accuracy (10-fold) | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| Decision Tree | 84.6% | 85.8% | 79.3% | 75.4% | 77.3% |
| Random Forest | 86.9% | 88.8% | 84.6% | 79.6% | 82.0% |
| Logistic Regression | 86.0% | 87.5% | 78.4% | 84.2% | 81.2% |
| **XGBoost** | **87.6%** | **89.0%** | **85.3%** | **79.6%** | **82.3%** |

Precision, recall and F1 are for the Dropout class. The best model, XGBoost, reaches 89.0% test accuracy. The strongest predictors are second- and first-semester approved units and grades, tuition fee status, and age at enrollment.

![Confusion matrix](confusion_matrix.png)
![Feature importance](feature_importance.png)

## How to run
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib ucimlrepo
python dropout_model.py data.csv
```
Outputs: `results.csv`, `confusion_matrix.png`, `feature_importance.png`.

## Files
- `dropout_model.py`: full pipeline (preprocessing, training, evaluation)
- `data.csv`: dataset
- `results.csv`: model comparison table
- `confusion_matrix.png`: confusion matrix of the best model
- `feature_importance.png`: top 10 predictors (Random Forest)

## Reference
Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). *Predict Students' Dropout and Academic Success.* UCI Machine Learning Repository.
