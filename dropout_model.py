"""Predicting Student Dropout in Higher Education Using Supervised ML Classification
Developers: Winnie Wasike, Naima Bashir, Mark Lemute
Dataset: UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2021)

Task: binary classification -> Dropout vs Non-Dropout (Enrolled + Graduate)

Usage:
    python dropout_model.py              # downloads data from UCI automatically
    python dropout_model.py data.csv     # or use a local CSV copy
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------------------- 1. LOAD DATA
if len(sys.argv) > 1:
    df = pd.read_csv(sys.argv[1], sep=None, engine="python", encoding="utf-8-sig")
else:
    from ucimlrepo import fetch_ucirepo          # pip install ucimlrepo
    d = fetch_ucirepo(id=697)
    df = d.data.features.copy()
    df["Target"] = d.data.targets.iloc[:, 0]
    df.to_csv("data.csv", index=False)           # keep a local copy for GitHub
df.columns = [c.strip() for c in df.columns]
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
df = df.drop_duplicates().dropna()

# ---------------------------------------------------------------- 2. TARGET
# Merge Enrolled + Graduate -> Non-Dropout (1 = Dropout)
y = (df["Target"] == "Dropout").astype(int)
X = df.drop(columns="Target")
print("Class distribution (1 = Dropout):\n", y.value_counts().rename({0: "Non-Dropout", 1: "Dropout"}), "\n")

# ---------------------------------------------------------------- 3. SPLIT
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {len(Xtr)} rows | Test: {len(Xte)} rows\n")

# ---------------------------------------------------------------- 4. MODELS
# Scaling + SMOTE live inside the pipeline, so they are fitted on training folds only (no leakage)
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1),
}
def make_pipe(clf):
    return Pipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])

# ---------------------------------------------------------------- 5. TRAIN + VALIDATE
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, clf in models.items():
    pipe = make_pipe(clf)
    cv_acc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    rows.append({
        "Model": name,
        "CV Accuracy (10-fold)": cv_acc.mean(),
        "Test Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred),
        "Recall": recall_score(yte, pred),
        "F1-score": f1_score(yte, pred),
    })
    fitted[name] = (pipe, pred)
    print(f"=== {name} ===")
    print(classification_report(yte, pred, target_names=["Non-Dropout", "Dropout"], zero_division=0))

res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("results.csv", index=False)

# ---------------------------------------------------------------- 6. BEST MODEL OUTPUTS
best = res.sort_values("F1-score", ascending=False).iloc[0]["Model"]
print("\nBest model:", best)
pipe, pred = fitted[best]
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["Non-Dropout", "Dropout"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

rf = fitted["Random Forest"][0].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\nTop 10 predictors (Random Forest):\n", imp.round(4))
imp[::-1].plot.barh(figsize=(8, 5), color="#3b6ea5")
plt.title("Top 10 Predictors of Dropout (Random Forest)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
print("\nSaved: results.csv, confusion_matrix.png, feature_importance.png")
