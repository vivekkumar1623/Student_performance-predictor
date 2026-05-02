"""
model_training.py
─────────────────
Train, evaluate, and save the Student Performance Predictor model.
Run this script independently to see full model evaluation report.

Usage:
    python model_training.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import LabelEncoder


# ─── 1. Generate Dataset ────────────────────────────────────────────────────
def generate_dataset(n=500, seed=42):
    np.random.seed(seed)

    study_hours     = np.random.randint(1, 10, n)
    attendance      = np.random.randint(50, 100, n)
    prev_score      = np.random.randint(30, 100, n)
    sleep_hours     = np.random.randint(4, 10, n)
    parental_edu    = np.random.choice(['None', 'High School', 'Bachelor', 'Master'], n)
    internet_access = np.random.choice(['Yes', 'No'], n)

    score = (study_hours * 4 + attendance * 0.3 + prev_score * 0.5 + sleep_hours * 2)
    label = (score > np.percentile(score, 40)).astype(int)

    df = pd.DataFrame({
        'Study_Hours_Per_Day': study_hours,
        'Attendance_Percent':  attendance,
        'Previous_Score':      prev_score,
        'Sleep_Hours':         sleep_hours,
        'Parental_Education':  parental_edu,
        'Internet_Access':     internet_access,
        'Result':              label
    })
    print(f"✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Pass: {df['Result'].sum()}  |  Fail: {(df['Result']==0).sum()}\n")
    return df


# ─── 2. Preprocess ──────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    le_edu = LabelEncoder()
    le_net = LabelEncoder()

    df['Parental_Education'] = le_edu.fit_transform(df['Parental_Education'])
    df['Internet_Access']    = le_net.fit_transform(df['Internet_Access'])

    X = df.drop('Result', axis=1)
    y = df['Result']
    return X, y, le_edu, le_net


# ─── 3. Compare Multiple Models ─────────────────────────────────────────────
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("=" * 50)
    print("     MODEL COMPARISON")
    print("=" * 50)

    results = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        cv  = cross_val_score(clf, X_train, y_train, cv=5).mean()
        results[name] = {"model": clf, "accuracy": acc, "cv_score": cv}
        print(f"  {name:<25} Accuracy: {acc*100:.2f}%   CV Score: {cv*100:.2f}%")

    print("=" * 50)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)\n")
    return results, best_name


# ─── 4. Detailed Evaluation ─────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n📊 CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
    print(f"  Model Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC Score  : {roc_auc:.4f}")

    return y_pred, y_prob, acc


# ─── 5. Plot Confusion Matrix ────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix – Random Forest')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.show()
    print("💾 Saved: confusion_matrix.png")


# ─── 6. Plot ROC Curve ──────────────────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Student Performance Predictor')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=100)
    plt.show()
    print("💾 Saved: roc_curve.png")


# ─── 7. Save Model ──────────────────────────────────────────────────────────
def save_model(model, le_edu, le_net):
    joblib.dump(model,  'rf_model.pkl')
    joblib.dump(le_edu, 'le_edu.pkl')
    joblib.dump(le_net, 'le_net.pkl')
    print("\n✅ Model saved as  →  rf_model.pkl")
    print("✅ Encoders saved  →  le_edu.pkl  |  le_net.pkl")


# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Student Performance Predictor – Model Training\n")

    # Step 1: Data
    df = generate_dataset()

    # Step 2: Preprocess
    X, y, le_edu, le_net = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📁 Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    # Step 3: Compare all models
    results, best_name = compare_models(X_train, X_test, y_train, y_test)

    # Step 4: Detailed evaluation of Random Forest
    rf_model = results["Random Forest"]["model"]
    y_pred, y_prob, acc = evaluate_model(rf_model, X_test, y_test)

    # Step 5: Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(rf_model, X_test, y_test)

    # Step 6: Save
    save_model(rf_model, le_edu, le_net)

    print(f"\n🎯 Final Accuracy: {acc*100:.2f}%")
    print("✅ Training complete!\n")
