# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  Project 2  : Data Classification Using AI
#  Algorithm  : K-Nearest Neighbors (KNN)
#  Dataset    : Iris Benchmark (150 samples | 3 classes | 4 features)
#  Pipeline   : Load → Scale → Split → Train → Evaluate → Predict
# ============================================================
#
#  HOW TO RUN:
#  pip install scikit-learn pandas numpy
#  python classifier.py
# ============================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)


# ── STEP 1: LOAD DATASET ─────────────────────────────────────
# Iris: 150 samples | 3 classes | 4 features
# Features: sepal_length, sepal_width, petal_length, petal_width
# Classes : Setosa (0), Versicolor (1), Virginica (2)

def load_data():
    iris       = load_iris()
    X          = pd.DataFrame(iris.data, columns=iris.feature_names)
    y          = pd.Series(iris.target, name="species")
    class_names = iris.target_names

    print("=" * 55)
    print("  PROJECT 2 — DATA CLASSIFICATION USING AI")
    print("  Algorithm : K-Nearest Neighbors (KNN)")
    print("  Dataset   : Iris Benchmark")
    print("=" * 55)
    print(f"\n[DATASET INFO]")
    print(f"  Samples     : {X.shape[0]}")
    print(f"  Features    : {X.shape[1]}")
    print(f"  Classes     : {list(class_names)}")
    print(f"\n[SAMPLE — First 5 rows]")
    print(X.head().to_string())
    print(f"\n[CLASS DISTRIBUTION]")
    dist = y.value_counts().rename(index=dict(enumerate(class_names)))
    print(dist.to_string())

    return X, y, class_names


# ── STEP 2: TRAIN-TEST SPLIT ─────────────────────────────────
# 80% training | 20% testing
# stratify=y → equal class representation in both splits
# shuffle=True → removes order bias (Iris is sorted by class)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = 0.2,
        random_state= 42,
        shuffle     = True,
        stratify    = y
    )
    print(f"\n[TRAIN-TEST SPLIT]")
    print(f"  Training samples : {len(X_train)}  (80%)")
    print(f"  Testing  samples : {len(X_test)}   (20%)")
    print(f"  Shuffle + Stratify: ✓ (order bias removed)")
    return X_train, X_test, y_train, y_test


# ── STEP 3: FEATURE SCALING (The Gatekeeper Rule) ────────────
# KNN uses Euclidean distance — unscaled large features dominate.
# StandardScaler: Mean = 0, Variance = 1
# CRITICAL: fit only on training data, transform both

def scale_features(X_train, X_test):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # learn scale from train
    X_test_scaled  = scaler.transform(X_test)        # apply same scale to test
    print(f"\n[FEATURE SCALING]")
    print(f"  StandardScaler applied → Mean = 0, Variance = 1")
    print(f"  Fitted on: training set only (no data leakage)")
    return X_train_scaled, X_test_scaled, scaler


# ── STEP 4: TRAIN KNN MODEL ──────────────────────────────────
# K = 5 : standard starting point (The Elbow heuristic)
# Proximity Principle: similar data points exist in close proximity
# Majority vote of K=5 nearest neighbors decides the class

def train_model(X_train_scaled, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    print(f"\n[MODEL TRAINING]")
    print(f"  Algorithm    : K-Nearest Neighbors")
    print(f"  K value      : {k}")
    print(f"  Status       : ✓ Trained successfully")
    return model


# ── STEP 5: EVALUATE MODEL ───────────────────────────────────
# Raw accuracy = "Accuracy Mirage" on imbalanced data
# We use F1 Score = Harmonic Mean of Precision and Recall
# Confusion Matrix = shows exactly where the model is confused

def evaluate_model(model, X_test_scaled, y_test, class_names):
    predictions = model.predict(X_test_scaled)

    acc    = accuracy_score(y_test, predictions)
    f1     = f1_score(y_test, predictions, average="weighted")
    cm     = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions,
        target_names=class_names
    )

    print(f"\n[MODEL EVALUATION]")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1 Score  : {f1 * 100:.2f}%  (Weighted Harmonic Mean)")

    print(f"\n[CONFUSION MATRIX]")
    cm_df = pd.DataFrame(
        cm,
        index   = [f"Actual:{c}"    for c in class_names],
        columns = [f"Pred:{c}"      for c in class_names]
    )
    print(cm_df.to_string())

    print(f"\n[CLASSIFICATION REPORT]")
    print(report)

    return predictions, acc, f1


# ── STEP 6: LIVE PREDICTION ──────────────────────────────────

def predict_new_sample(model, scaler, class_names):
    print(f"\n[LIVE PREDICTION]")
    print("  Enter new flower measurements to classify:\n")
    try:
        sl = float(input("  Sepal Length (cm) : "))
        sw = float(input("  Sepal Width  (cm) : "))
        pl = float(input("  Petal Length (cm) : "))
        pw = float(input("  Petal Width  (cm) : "))

        sample        = np.array([[sl, sw, pl, pw]])
        sample_scaled = scaler.transform(sample)
        pred          = model.predict(sample_scaled)[0]
        proba         = model.predict_proba(sample_scaled)[0]

        print(f"\n  → Predicted Species : {class_names[pred].upper()}")
        print(f"  → Confidence        : {proba[pred]*100:.1f}%")
        print(f"  → All Probabilities :")
        for i, name in enumerate(class_names):
            bar = "█" * int(proba[i] * 20)
            print(f"       {name:<14}: {proba[i]*100:5.1f}%  {bar}")

        return class_names[pred]

    except ValueError:
        print("  [!] Invalid input. Please enter numbers only.")
        return None


# ── MAIN RUNNER ───────────────────────────────────────────────

def run_classifier(interactive=True):
    """
    Full pipeline runner.
    interactive=False → skips live prediction (used by main.py pipeline).
    Returns: model, scaler, class_names
    """
    X, y, class_names            = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)
    model                            = train_model(X_train_sc, y_train, k=5)
    predictions, acc, f1             = evaluate_model(
                                         model, X_test_sc,
                                         y_test, class_names
                                       )

    if interactive:
        print("\n" + "-" * 55)
        choice = input("Classify a new flower sample? (y/n): ").strip().lower()
        if choice == "y":
            predict_new_sample(model, scaler, class_names)

    return model, scaler, class_names


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    run_classifier(interactive=True)
