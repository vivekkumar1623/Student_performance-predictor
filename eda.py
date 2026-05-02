"""
eda.py
──────
Exploratory Data Analysis (EDA) for the Student Performance Dataset.
Run this script to understand the data before model training.

Usage:
    python eda.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ─── 1. Generate Dataset (same as app.py) ───────────────────────────────────
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
    return df


# ─── 2. Basic Info ──────────────────────────────────────────────────────────
def basic_info(df):
    print("=" * 55)
    print("      STUDENT PERFORMANCE DATASET – EDA REPORT")
    print("=" * 55)

    print(f"\n📐 Shape           : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"🔍 Missing Values  : {df.isnull().sum().sum()}")
    print(f"📌 Duplicate Rows  : {df.duplicated().sum()}")

    print("\n📊 Class Distribution:")
    vc = df['Result'].value_counts()
    print(f"   Pass (1) → {vc[1]} students  ({vc[1]/len(df)*100:.1f}%)")
    print(f"   Fail (0) → {vc[0]} students  ({vc[0]/len(df)*100:.1f}%)")

    print("\n📈 Numerical Summary:")
    print(df.describe().round(2).to_string())

    print("\n🔤 Categorical Columns:")
    for col in ['Parental_Education', 'Internet_Access']:
        print(f"\n  [{col}]")
        for val, cnt in df[col].value_counts().items():
            print(f"    {val:<15} : {cnt}")


# ─── 3. Correlation Heatmap ──────────────────────────────────────────────────
def plot_correlation(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True)
    plt.title("Correlation Heatmap – Numerical Features", fontsize=13)
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=100)
    plt.show()
    print("💾 Saved: eda_correlation.png")


# ─── 4. Distribution of Each Feature ────────────────────────────────────────
def plot_distributions(df):
    num_cols = ['Study_Hours_Per_Day', 'Attendance_Percent',
                'Previous_Score', 'Sleep_Hours']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue='Result',
                     kde=True, ax=axes[i],
                     palette={1: '#4CAF50', 0: '#F44336'},
                     legend=(i == 0))
        axes[i].set_title(f"Distribution of {col.replace('_', ' ')}")
        axes[i].set_xlabel(col.replace('_', ' '))

    if i == 0:
        axes[0].legend(title='Result', labels=['Fail', 'Pass'])

    plt.suptitle("Feature Distributions by Pass/Fail", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=100)
    plt.show()
    print("💾 Saved: eda_distributions.png")


# ─── 5. Categorical Feature vs Result ───────────────────────────────────────
def plot_categorical(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parental Education
    edu_order = ['None', 'High School', 'Bachelor', 'Master']
    edu_data = (
        df.groupby(['Parental_Education', 'Result'])
          .size()
          .reset_index(name='Count')
    )
    sns.barplot(data=edu_data, x='Parental_Education', y='Count',
                hue='Result', order=edu_order, ax=axes[0],
                palette={1: '#4CAF50', 0: '#F44336'})
    axes[0].set_title("Parental Education vs Result")
    axes[0].set_xlabel("Parental Education Level")
    axes[0].legend(title='Result', labels=['Fail', 'Pass'])

    # Internet Access
    net_data = (
        df.groupby(['Internet_Access', 'Result'])
          .size()
          .reset_index(name='Count')
    )
    sns.barplot(data=net_data, x='Internet_Access', y='Count',
                hue='Result', ax=axes[1],
                palette={1: '#4CAF50', 0: '#F44336'})
    axes[1].set_title("Internet Access vs Result")
    axes[1].set_xlabel("Has Internet Access")
    axes[1].legend(title='Result', labels=['Fail', 'Pass'])

    plt.suptitle("Categorical Features vs Student Result", fontsize=14)
    plt.tight_layout()
    plt.savefig("eda_categorical.png", dpi=100)
    plt.show()
    print("💾 Saved: eda_categorical.png")


# ─── 6. Boxplots – Outlier Check ─────────────────────────────────────────────
def plot_boxplots(df):
    num_cols = ['Study_Hours_Per_Day', 'Attendance_Percent',
                'Previous_Score', 'Sleep_Hours']

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x='Result', y=col,
                    palette={1: '#4CAF50', 0: '#F44336'},
                    ax=axes[i])
        axes[i].set_title(col.replace('_', '\n'))
        axes[i].set_xticklabels(['Fail', 'Pass'])

    plt.suptitle("Boxplots: Feature Spread for Pass vs Fail", fontsize=13)
    plt.tight_layout()
    plt.savefig("eda_boxplots.png", dpi=100)
    plt.show()
    print("💾 Saved: eda_boxplots.png")


# ─── 7. Key Insights Summary ─────────────────────────────────────────────────
def print_insights(df):
    pass_df = df[df['Result'] == 1]
    fail_df = df[df['Result'] == 0]

    print("\n" + "=" * 55)
    print("          💡 KEY INSIGHTS")
    print("=" * 55)

    for col in ['Study_Hours_Per_Day', 'Attendance_Percent',
                'Previous_Score', 'Sleep_Hours']:
        p_mean = pass_df[col].mean()
        f_mean = fail_df[col].mean()
        diff   = p_mean - f_mean
        print(f"\n  {col.replace('_', ' '):<25}")
        print(f"    Passing students avg : {p_mean:.2f}")
        print(f"    Failing students avg : {f_mean:.2f}")
        print(f"    Difference           : +{diff:.2f}" if diff > 0 else f"    Difference : {diff:.2f}")

    internet_pass_rate = pass_df['Internet_Access'].value_counts(normalize=True).get('Yes', 0)
    print(f"\n  Internet Access (Yes) in passing students: {internet_pass_rate*100:.1f}%")
    print("=" * 55)


# ─── MAIN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Running Exploratory Data Analysis...\n")

    df = generate_dataset()

    basic_info(df)
    plot_correlation(df)
    plot_distributions(df)
    plot_categorical(df)
    plot_boxplots(df)
    print_insights(df)

    print("\n✅ EDA complete! All charts saved as PNG files.\n")
