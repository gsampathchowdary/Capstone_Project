# Module 2: Analytics Pipeline (`/analytics`)

# Executive Summary
This module provides a unified data profiling, exploratory analysis, and predictive modeling pipeline on the Titanic dataset, simulating customer survival and delivery compliance modeling for Zepto's analytics guild.

---

# Part A: Data Profiling & Data Story

# 1. Missing Value Strategy
- **`deck` (77.10% missing > 30% threshold)**: Column dropped entirely due to unreliable imputation.
- **`age` (19.87% missing -> 5%-30% threshold)**: Imputed using median age (28.0 years).
- **`embarked` / `embark_town` (0.22% missing < 5% threshold)**: Imputed using mode (`S` / `Southampton`).

# 2. Univariate Analysis & Outlier Detection
- **Age**: Q1 = 22.0, Q3 = 35.0, IQR = 13.0. Outliers = 66 records outside [2.5, 54.5].
- **Fare**: Q1 = 7.91, Q3 = 31.00, IQR = 23.09. Outliers = 116 records outside [-26.72, 65.63].
- **Fare Skewness**: Mean ($32.20) > Median ($14.45) > Mode ($8.05). **Conclusion**: Strongly right-skewed.

# 3. Bivariate Analysis & Correlation Matrix
- **Survival Rate by Sex**: Female = **74.20%**, Male = **18.89%**.
- **Survival Rate by Pclass**: 1st Class = **62.96%**, 2nd Class = **47.28%**, 3rd Class = **24.24%**.
- **Top 2 Strongest Off-Diagonal Correlations**:
  1. `(pclass, fare)`: $r = -0.55$ (Higher classes paid significantly higher fares).
  2. `(sibsp, parch)`: $r = +0.41$ (Passengers traveling with siblings/spouses also traveled with parents/children).

# 4. Multivariate Data Story (4 Charts)
1. **Chart 1 (Sex & Pclass)**: Females in 1st/2nd class achieved >92% survival, confirming maritime evacuation priority.
2. **Chart 2 (Fare & Survival)**: Survived passengers consistently exhibit higher median fares across all classes.
3. **Chart 3 (Age vs Fare Scatter)**: High fare ticket holders in 1st class survived regardless of age.
4. **Chart 4 (Family Size)**: Family sizes of 2–4 members achieved the highest survival rates (~55–72%), while solo travelers and large families (>5) suffered higher mortality.

---

# Part B: Machine Learning & Evaluation

# Master Model Comparison Table

| Model Type | Model / Strategy | Accuracy / MAE | Precision / RMSE | Recall / R² | F1 Score / Adj R² | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classifier** | Logistic Regression | 0.8045 | 0.7727 | 0.7246 | 0.7478 | 0.8524 |
| **Classifier** | Decision Tree | 0.7933 | 0.8039 | 0.5942 | 0.6833 | 0.8351 |
| **Classifier** | **Random Forest (Tuned)** | **0.8268** | **0.8197** | **0.7246** | **0.7692** | **0.8681** |
| **Regression** | Linear Regression (Fare) | MAE: $19.12 | RMSE: $34.58 | R²: 0.3812 | Adj R²: 0.3521 | N/A |

# Class Imbalance Strategy Comparison
- **Baseline Random Forest**: F1 = 0.7519
- **Class Weight Balanced**: F1 = 0.7594
- **SMOTE (Train Fold Only)**: F1 = **0.7619** (Optimal balance between Precision and Recall).

# GridSearch & OOB Score
- **Best Hyperparameters**: `max_depth: 6`, `max_features: 'sqrt'`, `n_estimators: 100`.
- **Out-of-Bag (OOB) Score**: **0.8160** (81.60% internal validation accuracy).

# Regression Heteroscedasticity Conclusion
The residual plot for Fare prediction demonstrates clear **heteroscedasticity** (a non-constant, fan-shaped residual spread), showing that variance increases significantly for high-value luxury tickets.

# Final Deployment Recommendation
We recommend deploying the **Tuned Random Forest Classifier Pipeline** (`full_zepto_titanic_pipeline.pkl`). It achieved the top F1-Score (**0.7692**) and ROC-AUC (**0.8681**), demonstrating balanced precision and recall on unseen test data without overfitting.