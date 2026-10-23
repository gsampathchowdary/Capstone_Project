# Module 2: Analytics, Machine Learning & PyTorch Deep Embeddings Report

## 1. Exploratory Visualizations

### Univariate Distributions: Age and Fare
![Age and Fare Distributions](../charts/age_fare_distributions.png)

### 6x6 Correlation Heatmap
![Correlation Heatmap](../charts/correlation_heatmap.png)

### Multivariate Data Story Charts
| Chart 1: Survival by Class and Sex | Chart 2: Fare by Class and Survival |
|---|---|
| ![Chart 1](../charts/chart1_survival_sex_pclass.png) | ![Chart 2](../charts/chart2_fare_by_class_survival.png) |
| Chart 3: Age vs Fare Scatter | Chart 4: Family Size Survival |
| ![Chart 3](../charts/chart3_age_vs_fare_scatter.png) | ![Chart 4](../charts/chart4_family_size_survival.png) |

## 2. Predictive Modeling & Classification Outputs

### Decision Tree Structure
![Decision Tree](../charts/decision_tree.png)

### Fare Regression Residual Plot
![Residual Plot](../charts/fare_regression_residuals.png)

### Classifier Performance Summary

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7821 | 0.8409 | 0.5362 | 0.6549 | 0.807 |
| Random Forest | 0.7989 | 0.7797 | 0.6667 | 0.7188 | 0.8294 |

### 3. PyTorch Deep Neural Embeddings & Unsupervised Segmentation

- **Embedding Latent Dimension**: 16
- **Autoencoder Training Epochs**: 50
- **Final Reconstruction Loss (MSE)**: 0.332552
- **Identified Customer Clusters**: 4
- **Artifacts Saved**: `models/pytorch_customer_autoencoder.pt`, `outputs/segmented_customers_output.csv`

### 4. Regression Side-Task: Fare Prediction

- Mean Absolute Error: 20.81
- Root Mean Squared Error: 30.47
- R-Squared: 0.3999
- Adjusted R-Squared: 0.3604
