import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TeeLogger:
    """Mirrors console prints directly into an output log file."""
    def __init__(self, filepath, mode="w"):
        self.terminal = sys.stdout
        self.logfile = open(filepath, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def step1_load_and_save_data():
    """Loads dataset from Seaborn cache once and saves offline fallback CSV."""
    print("=========================================================")
    print("      TASK 1: DATA LOADING & OFFLINE FALLBACK CREATION   ")
    print("=========================================================")

    df = sns.load_dataset("titanic")
    df.to_csv(CSV_PATH, index=False)

    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Saved offline fallback dataset to '{CSV_PATH}'\n")
    return df


def step2_profiling_and_missing_values(df):
    """Profiles dataset and applies percentage-based threshold rules for missing values."""
    print("=========================================================")
    print("      TASK 2: PROFILING & MISSING-VALUE ANALYSIS        ")
    print("=========================================================")

    print("\n--- DataFrame Info ---")
    df.info()

    print("\n--- Summary Statistics ---")
    print(df.describe().to_string())

    missing_series = df.isnull().mean() * 100
    missing_cols = missing_series[missing_series > 0].round(2)

    print("\n--- Missing Value Percentages per Affected Column ---")
    for col, pct in missing_cols.items():
        print(f" - {col}: {pct}% missing")

    print("\n--- Missing Value Strategy Decisions ---")
    print("1. deck [77.10% missing > 30% threshold]: Dropped column because missing rate is too high for reliable imputation.")
    print("2. age [19.87% missing, within 5% to 30% threshold]: Imputed with median age.")
    print("3. embarked & embark_town [0.22% missing < 5% threshold]: Imputed with mode value.\n")

    df_clean = df.copy()
    df_clean.drop(columns=["deck"], inplace=True)
    df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
    df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])
    df_clean["embark_town"] = df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0])

    return df_clean


def step3_univariate_analysis(df):
    """Computes IQR outliers and skewness for Age and Fare."""
    print("=========================================================")
    print("      TASK 3: UNIVARIATE ANALYSIS [AGE & FARE]           ")
    print("=========================================================")

    for col in ["age", "fare"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\n--- {col.upper()} Statistics & Outliers ---")
        print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
        print(f"Outlier Range Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"Total Outlier Count: {len(outliers)}")

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode()[0]

    print("\n--- FARE Central Tendency & Skewness ---")
    print(f"Mean: {fare_mean:.2f} | Median: {fare_median:.2f} | Mode: {fare_mode:.2f}")
    print(f"Ordering: Mean {fare_mean:.2f} > Median {fare_median:.2f} > Mode {fare_mode:.2f}")
    print("Distribution Shape: Fare is strongly right-skewed with positive skew.\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="skyblue").set(title="Age Distribution")
    sns.boxplot(x=df["age"], ax=axes[0, 1], color="skyblue").set(title="Age Boxplot")
    sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="salmon").set(title="Fare Distribution")
    sns.boxplot(x=df["fare"], ax=axes[1, 1], color="salmon").set(title="Fare Boxplot")
    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "age_fare_distributions.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"[SAVED] Univariate chart saved to: {chart_path}")


def step4_bivariate_analysis(df):
    """Computes survival rates using boolean masks and plots the 6x6 numeric correlation matrix."""
    print("=========================================================")
    print("      TASK 4: BIVARIATE ANALYSIS & CORRELATION MATRIX    ")
    print("=========================================================")

    sr_female = df[df["sex"] == "female"]["survived"].mean() * 100
    sr_male = df[df["sex"] == "male"]["survived"].mean() * 100
    print(f"a. Survival Rate by Sex: Female = {sr_female:.2f}%, Male = {sr_male:.2f}%")

    print("\nb. Survival Rate by Pclass:")
    for pclass in [1, 2, 3]:
        rate = df[df["pclass"] == pclass]["survived"].mean() * 100
        print(f" - Pclass {pclass}: {rate:.2f}%")

    print("\nc. Survival Rate by Sex and Pclass:")
    for sex in ["female", "male"]:
        for pclass in [1, 2, 3]:
            mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            rate = df[mask]["survived"].mean() * 100
            print(f" - {sex.capitalize()}, Pclass {pclass}: {rate:.2f}%")

    # 6x6 numeric correlation matrix excluding adult_male and alone
    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[numeric_cols].corr()

    print("\n--- 6x6 Numeric Feature Correlation Matrix ---")
    print(corr_matrix.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("6x6 Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(CHARTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"\n[SAVED] Correlation heatmap saved to: {heatmap_path}")

    print("\n--- Top 2 Strongest Off-Diagonal Correlations ---")
    print("1. pclass and fare: r = -0.55. Strong negative correlation where 1st class passengers paid substantially higher fares.")
    print("2. sibsp and parch: r = +0.41. Moderate positive correlation where passengers traveling with siblings or spouses were often accompanied by parents or children.\n")


def step5_multivariate_data_story(df):
    """Generates 4 distinct multivariate charts."""
    print("=========================================================")
    print("      TASK 5: MULTIVARIATE DATA STORY CHARTS             ")
    print("=========================================================")

    # Chart 1: Survival Rate by Sex and Pclass
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None, palette="Set2")
    plt.title("Chart 1: Survival Rate Breakdown by Pclass and Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart1_survival_sex_pclass.png"))
    plt.close()

    # Chart 2: Fare Distribution by Pclass and Survival Status
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived", palette="Set1")
    plt.ylim(0, 300)
    plt.title("Chart 2: Fare Distribution by Pclass and Survival Status")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart2_fare_by_class_survival.png"))
    plt.close()

    # Chart 3: Age vs Fare Scatter Plot colored by Survival
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.7, palette="coolwarm")
    plt.ylim(0, 300)
    plt.title("Chart 3: Age vs Fare Scatter Plot by Survival Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart3_age_vs_fare_scatter.png"))
    plt.close()

    # Chart 4: Family Size vs Survival Rate
    df_copy = df.copy()
    df_copy["family_size"] = df_copy["sibsp"] + df_copy["parch"] + 1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_copy, x="family_size", y="survived", palette="viridis", errorbar=None)
    plt.title("Chart 4: Survival Rate by Family Size")
    plt.xlabel("Family Size [SibSp + ParCh + 1]")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart4_family_size_survival.png"))
    plt.close()

    print(f"[SAVED] All 4 multivariate charts saved into: '{CHARTS_DIR}'.\n")


def step6_zscore_standardization_check(df):
    """Performs exploratory Z-score sanity check on Age and Fare."""
    print("=========================================================")
    print("      TASK 6: EDA STANDARDIZATION SANITY CHECK          ")
    print("=========================================================")

    df_std = df.copy()
    for col in ["age", "fare"]:
        mean_orig = df_std[col].mean()
        std_orig = df_std[col].std()
        df_std[f"{col}_zscore"] = (df_std[col] - mean_orig) / std_orig

        print(f"\n--- {col.upper()} Z-Score Standardization ---")
        print(f"Original Mean: {mean_orig:.4f} | Original Std: {std_orig:.4f}")
        print(f"Z-Score Mean : {df_std[f'{col}_zscore'].mean():.4f} [approximately 0.0]")
        print(f"Z-Score Std  : {df_std[f'{col}_zscore'].std():.4f} [approximately 1.0]")

    print("\nNote: This standardization is strictly an exploratory check and is not leaked into the predictive modeling pipeline.\n")


if __name__ == "__main__":
    log_path = os.path.join(OUTPUT_DIR, "analytics_output.txt")
    logger = TeeLogger(log_path, mode="w")
    sys.stdout = logger

    try:
        raw_df = step1_load_and_save_data()
        clean_df = step2_profiling_and_missing_values(raw_df)
        step3_univariate_analysis(clean_df)
        step4_bivariate_analysis(clean_df)
        step5_multivariate_data_story(clean_df)
        step6_zscore_standardization_check(clean_df)
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"[OUTPUT SAVED] Module 2 Part A EDA logs saved to: {log_path}")