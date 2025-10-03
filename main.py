#!.venv/bin/python3

# PCA trade-offs demo on the Iris dataset
# - Baseline Logistic Regression (all features)
# - PCA to 2 components + Logistic Regression
# - Simple feature selection (drop 1 feature) + Logistic Regression
# Plots:
#   1) Test set in PCA space (PC1 vs PC2)
#   2) Explained variance ratio for the top 2 PCs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------
# 1) Load and split data
# ---------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
print("\nIris dataset loaded:")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# 2) Standardize features
# ---------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------
# 3) Baseline model (no PCA)
# ---------------------------
logreg_base = LogisticRegression(max_iter=200)
logreg_base.fit(X_train_s, y_train)
y_pred_base = logreg_base.predict(X_test_s)
acc_base = accuracy_score(y_test, y_pred_base)

# ---------------------------
# 4) PCA to 2 components
# ---------------------------
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

logreg_pca2 = LogisticRegression(max_iter=200)
logreg_pca2.fit(X_train_pca, y_train)
y_pred_pca2 = logreg_pca2.predict(X_test_pca)
acc_pca2 = accuracy_score(y_test, y_pred_pca2)

explained = pca.explained_variance_ratio_

# ---------------------------
# 5) Simple feature selection: drop one feature
# ---------------------------
drop_col = 'petal width (cm)'
X_train_sel = X_train.drop(columns=[drop_col])
X_test_sel = X_test.drop(columns=[drop_col])

scaler_sel = StandardScaler()
X_train_sel_s = scaler_sel.fit_transform(X_train_sel)
X_test_sel_s = scaler_sel.transform(X_test_sel)

logreg_sel = LogisticRegression(max_iter=200)
logreg_sel.fit(X_train_sel_s, y_train)
y_pred_sel = logreg_sel.predict(X_test_sel_s)
acc_sel = accuracy_score(y_test, y_pred_sel)

# ---------------------------
# 6) Summarize results
# ---------------------------
results = pd.DataFrame({
    "Setup": [
        "Baseline (all 4 features)",
        "PCA (2 components)",
        f"Feature selection (drop '{drop_col}')"
    ],
    "Test Accuracy": [acc_base, acc_pca2, acc_sel]
}).sort_values("Test Accuracy", ascending=False)

print("\n=== PCA vs Baseline vs Feature Selection – Iris ===")
print(results.to_string(index=False))

print("\nExplained variance ratio (PCA 2 components):")
for i, r in enumerate(explained, start=1):
    print(f"  PC{i}: {r:.3f}")
print(f"  Total (PC1+PC2): {explained.sum():.3f}")

# ---------------------------
# 7) Plot: Test set in PCA space
# ---------------------------
plt.figure()
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test.to_numpy())
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris – Test Set in PCA Space (2D)")
plt.tight_layout()
plt.show()

# ---------------------------
# 8) Plot: Explained variance ratio
# ---------------------------
plt.figure()
components = np.arange(1, len(explained) + 1)
plt.bar(components, explained)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by PC (top 2)")
plt.xticks(components)
plt.tight_layout()
plt.show()
