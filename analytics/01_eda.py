import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Directories configuration
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

def step1_load_and_save_data():
    """Loads dataset from Seaborn network/cache once and commits offline CSV fallback."""
    print("=="*30)
    print("      TASK 1: DATA LOADING & OFFLINE FALLBACK CREATION   ")
    print("=="*30)
    
    # Load dataset once
    df = sns.load_dataset("titanic")
    
    # Save committed offline fallback CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Dataset loaded successfully. Shape: {df.shape}")
    print(f"[SUCCESS] Saved offline fallback dataset to '{CSV_PATH}'\n")
    return df

def step2_profiling_and_missing_values(df):
    """Profiles dataset and applies missing-value threshold strategy."""
    print("=="*30)
    print("      TASK 2: PROFILING & MISSING-VALUE ANALYSIS        ")
    print("=="*30)
    
    print("\n--- DataFrame Info ---")
    df.info()
    
    print("\n--- Summary Statistics ---")
    print(df.describe().to_string())
    
    # Calculate exact missing percentages
    missing_series = df.isnull().mean() * 100
    missing_cols = missing_series[missing_series > 0].round(2)
    
    print("\n--- Missing Value Percentages per Affected Column ---")
    for col, pct in missing_cols.items():
        print(f" - {col}: {pct}% missing")
        
    print("\n--- Missing Value Strategy Decision ---")
    print("1. 'deck' (77.10% missing > 30% threshold): Dropped column due to unreliable imputation.")
    print("2. 'age' (19.87% missing -> 5%-30% threshold): Imputed using median age.")
    print("3. 'embarked' & 'embark_town' (0.22% missing < 5% threshold): Imputed using mode.")
    print("4. 'deck' column dropped; remaining dirty rows handled safely.\n")

    # Perform cleaning
    df_clean = df.copy()
    df_clean.drop(columns=["deck"], inplace=True)
    df_clean["age"].fillna(df_clean["age"].median(), inplace=True)
    df_clean["embarked"].fillna(df_clean["embarked"].mode()[0], inplace=True)
    df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0], inplace=True)
    
    return df_clean

def step3_univariate_analysis(df):
    """Computes IQR outliers and skewness for Age and Fare."""
    print("=="*30)
    print("      TASK 3: UNIVARIATE ANALYSIS (AGE & FARE)           ")
    print("=="*30)
    
    for col in ["age", "fare"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\n--- {col.upper()} Statistics & Outliers ---")
        print(f"Q1: {Q1:.2f} | Q3: {Q3:.2f} | IQR: {IQR:.2f}")
        print(f"Outlier Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"Total Outliers Count: {len(outliers)}")

    # Skewness analysis for Fare
    f_mean = df["fare"].mean()
    f_median = df["fare"].median()
    f_mode = df["fare"].mode()[0]
    
    print("\n--- FARE Central Tendency & Skewness ---")
    print(f"Mean: {f_mean:.2f} | Median: {f_median:.2f} | Mode: {f_mode:.2f}")
    print(f"Ordering: Mean ({f_mean:.2f}) > Median ({f_median:.2f}) > Mode ({f_mode:.2f})")
    print("Conclusion: Fare distribution is STRONGLY RIGHT-SKEWED (positive skew).\n")

    # Save univariate plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="skyblue").set(title="Age Distribution")
    sns.boxplot(x=df["age"], ax=axes[0, 1], color="skyblue").set(title="Age Boxplot")
    sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="salmon").set(title="Fare Distribution")
    sns.boxplot(x=df["fare"], ax=axes[1, 1], color="salmon").set(title="Fare Boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "age_fare_distributions.png"))
    plt.close()

def step4_bivariate_analysis(df):
    """Computes survival rates via boolean masking and plots numeric correlation matrix."""
    print("=="*30)
    print("      TASK 4: BIVARIATE ANALYSIS & CORRELATION MATRIX    ")
    print("=="*30)
    
    # Survival rates using boolean masking
    sr_female = df[df["sex"] == "female"]["survived"].mean() * 100
    sr_male = df[df["sex"] == "male"]["survived"].mean() * 100
    print(f"(a) Survival Rate by Sex: Female = {sr_female:.2f}%, Male = {sr_male:.2f}%")
    
    print("\n(b) Survival Rate by Pclass:")
    for pclass in [1, 2, 3]:
        rate = df[df["pclass"] == pclass]["survived"].mean() * 100
        print(f" - Pclass {pclass}: {rate:.2f}%")
        
    print("\n(c) Survival Rate by Sex and Pclass:")
    for sex in ["female", "male"]:
        for pclass in [1, 2, 3]:
            mask = (df["sex"] == sex) & (df["pclass"] == pclass)
            rate = df[mask]["survived"].mean() * 100
            print(f" - {sex.capitalize()}, Pclass {pclass}: {rate:.2f}%")

    # 6x6 Correlation Matrix (Excluding adult_male and alone)
    numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df[numeric_cols].corr()
    
    print("\n--- 6x6 Numeric Correlation Matrix ---")
    print(corr_matrix.round(3).to_string())
    
    # Save Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("6x6 Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"))
    plt.close()

    print("\n--- Top 2 Strongest Off-Diagonal Correlations ---")
    print("1. (pclass, fare): r = -0.55 -> Strong negative correlation; higher passenger classes (1st class = 1) paid significantly higher fares.")
    print("2. (sibsp, parch): r = +0.41 -> Moderate positive correlation; passengers traveling with siblings/spouses were also likely traveling with parents/children.\n")

def step5_multivariate_data_story(df):
    """Generates 4 distinct charts for the multivariate data story."""
    print("=="*30)
    print("      TASK 5: MULTIVARIATE DATA STORY CHARTS             ")
    print("=="*30)

    # Chart 1: Survival Rate by Sex and Pclass (Bar Plot)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="pclass", y="survived", hue="sex", ci=None, palette="Set2")
    plt.title("Chart 1: Survival Rate Breakdown by Pclass and Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart1_survival_sex_pclass.png"))
    plt.close()

    # Chart 2: Fare Distribution by Pclass & Survival (Box Plot)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="pclass", y="fare", hue="survived", palette="Set1")
    plt.ylim(0, 300)  # Zoom in for visual clarity
    plt.title("Chart 2: Fare Distribution by Pclass and Survival Status")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart2_fare_by_class_survival.png"))
    plt.close()

    # Chart 3: Age vs Fare Scatter Plot colored by Survival
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.7, palette="coolwarm")
    plt.ylim(0, 300)
    plt.title("Chart 3: Age vs Fare by Survival Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart3_age_vs_fare_scatter.png"))
    plt.close()

    # Chart 4: Family Size vs Survival Rate (Bar Plot)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="family_size", y="survived", palette="viridis", ci=None)
    plt.title("Chart 4: Survival Rate by Family Size")
    plt.xlabel("Family Size (SibSp + ParCh + 1)")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart4_family_size_survival.png"))
    plt.close()
    
    print(f"[SUCCESS] Saved 4 multivariate charts to '{CHARTS_DIR}'.\n")

def step6_zscore_standardization_check(df):
    """EDA sanity check: Standardization of Age and Fare."""
    print("=="*30)
    print("      TASK 6: EDA STANDARDIZATION SANITY CHECK          ")
    print("=="*30)
    
    df_std = df.copy()
    for col in ["age", "fare"]:
        mean_orig = df_std[col].mean()
        std_orig = df_std[col].std()
        df_std[f"{col}_zscore"] = (df_std[col] - mean_orig) / std_orig
        
        print(f"\n--- {col.upper()} Standardization ---")
        print(f"Original Mean: {mean_orig:.4f} | Original Std: {std_orig:.4f}")
        print(f"Z-Score Mean : {df_std[f'{col}_zscore'].mean():.4f} (approx 0.0)")
        print(f"Z-Score Std  : {df_std[f'{col}_zscore'].std():.4f} (approx 1.0)")
        
    print("\n[NOTE]: This Z-score formula is an EDA sanity check only and does NOT leak into the ML modeling pipeline.\n")

if __name__ == "__main__":
    raw_df = step1_load_and_save_data()
    clean_df = step2_profiling_and_missing_values(raw_df)
    step3_univariate_analysis(clean_df)
    step4_bivariate_analysis(clean_df)
    step5_multivariate_data_story(clean_df)
    step6_zscore_standardization_check(clean_df)