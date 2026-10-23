import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TeeLogger:
    """Mirrors console prints directly into an output log file."""
    def __init__(self, filepath, mode="a"):
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


def load_data():
    """Loads dataset from committed offline fallback titanic.csv."""
    df = pd.read_csv(CSV_PATH)
    if "deck" in df.columns:
        df.drop(columns=["deck"], inplace=True)
    return df


class CustomerEmbeddingAutoencoder(nn.Module):
    """
    Custom PyTorch Deep Neural Autoencoder to extract dense latent embeddings
    for unsupervised customer segmentation and propensity modeling.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def generate_markdown_report(metrics_list, reg_metrics, pytorch_summary):
    """Generates a structured markdown file linking all charts, evaluation metrics, and PyTorch embeddings summary."""
    report_path = os.path.join(OUTPUT_DIR, "analytics_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Module 2: Analytics, Machine Learning & PyTorch Deep Embeddings Report\n\n")
        f.write("## 1. Exploratory Visualizations\n\n")
        f.write("### Univariate Distributions: Age and Fare\n")
        f.write("![Age and Fare Distributions](../charts/age_fare_distributions.png)\n\n")
        f.write("### 6x6 Correlation Heatmap\n")
        f.write("![Correlation Heatmap](../charts/correlation_heatmap.png)\n\n")
        f.write("### Multivariate Data Story Charts\n")
        f.write("| Chart 1: Survival by Class and Sex | Chart 2: Fare by Class and Survival |\n")
        f.write("|---|---|\n")
        f.write("| ![Chart 1](../charts/chart1_survival_sex_pclass.png) | ![Chart 2](../charts/chart2_fare_by_class_survival.png) |\n")
        f.write("| Chart 3: Age vs Fare Scatter | Chart 4: Family Size Survival |\n")
        f.write("| ![Chart 3](../charts/chart3_age_vs_fare_scatter.png) | ![Chart 4](../charts/chart4_family_size_survival.png) |\n\n")
        
        f.write("## 2. Predictive Modeling & Classification Outputs\n\n")
        f.write("### Decision Tree Structure\n")
        f.write("![Decision Tree](../charts/decision_tree.png)\n\n")
        f.write("### Fare Regression Residual Plot\n")
        f.write("![Residual Plot](../charts/fare_regression_residuals.png)\n\n")
        f.write("### Classifier Performance Summary\n\n")
        f.write("| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in metrics_list:
            f.write(f"| {m['Model']} | {m['Accuracy']} | {m['Precision']} | {m['Recall']} | {m['F1 Score']} | {m['ROC-AUC']} |\n")
        
        f.write("\n### 3. PyTorch Deep Neural Embeddings & Unsupervised Segmentation\n\n")
        f.write(f"- **Embedding Latent Dimension**: {pytorch_summary['embedding_dim']}\n")
        f.write(f"- **Autoencoder Training Epochs**: {pytorch_summary['epochs']}\n")
        f.write(f"- **Final Reconstruction Loss (MSE)**: {pytorch_summary['final_loss']:.6f}\n")
        f.write(f"- **Identified Customer Clusters**: {pytorch_summary['n_clusters']}\n")
        f.write(f"- **Artifacts Saved**: `models/pytorch_customer_autoencoder.pt`, `outputs/segmented_customers_output.csv`\n\n")

        f.write("### 4. Regression Side-Task: Fare Prediction\n\n")
        f.write(f"- Mean Absolute Error: {reg_metrics['MAE']}\n")
        f.write(f"- Root Mean Squared Error: {reg_metrics['RMSE']}\n")
        f.write(f"- R-Squared: {reg_metrics['R2']}\n")
        f.write(f"- Adjusted R-Squared: {reg_metrics['Adj R2']}\n")
    
    print(f"\n[REPORT GENERATED] Visual Markdown summary saved at: {report_path}")


def run_modeling_pipeline():
    """Executes classification modeling, PyTorch deep neural embeddings, regression side-task, and exports reports."""
    df = load_data()
    
    # -------------------------------------------------------------
    # TASK 7: STRATIFIED TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    print("=========================================================")
    print("            TASK 7: STRATIFIED TRAIN/TEST SPLIT          ")
    print("=========================================================")
    
    X = df.drop(columns=["survived", "alive"])
    y = df["survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Train survival mean: {y_train.mean():.4f} | Test survival mean: {y_test.mean():.4f}\n")

    # -------------------------------------------------------------
    # TASK 8: PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    num_features = ["age", "fare", "pclass", "sibsp", "parch"]
    cat_features = ["sex", "embarked"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # -------------------------------------------------------------
    # TASK 9 & 10: TRAIN AND EVALUATE 3 CLASSIFIERS
    # -------------------------------------------------------------
    print("=========================================================")
    print("      TASK 9 & 10: CLASSIFIER TRAINING & EVALUATION      ")
    print("=========================================================")

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    }

    results = []

    for name, clf in classifiers.items():
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # Save Decision Tree Visualization
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    
    cat_onehot_cols = dt_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_features).tolist()
    feature_names = num_features + cat_onehot_cols

    plt.figure(figsize=(14, 8))
    plot_tree(
        dt_pipeline.named_steps["classifier"],
        feature_names=feature_names,
        class_names=["Died", "Survived"],
        filled=True,
        fontsize=10
    )
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "decision_tree.png"))
    plt.close()
    print(f"\n[SAVED] Decision tree chart saved to '{CHARTS_DIR}/decision_tree.png'.\n")

    # -------------------------------------------------------------
    # TASK 11: IMBALANCE HANDLING COMPARISON
    # -------------------------------------------------------------
    print("=========================================================")
    print("        TASK 11: CLASS IMBALANCE STRATEGY COMPARISON      ")
    print("=========================================================")

    rf_base = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
    rf_base.fit(X_train, y_train)
    p_base = rf_base.predict(X_test)

    rf_bal = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
    rf_bal.fit(X_train, y_train)
    p_bal = rf_bal.predict(X_test)

    rf_smote = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_smote.fit(X_train, y_train)
    p_smote = rf_smote.predict(X_test)

    print("Imbalance Handling Results:")
    print(f"1. Baseline       -> Precision: {precision_score(y_test, p_base):.4f}, Recall: {recall_score(y_test, p_base):.4f}, F1: {f1_score(y_test, p_base):.4f}")
    print(f"2. Class Balanced -> Precision: {precision_score(y_test, p_bal):.4f}, Recall: {recall_score(y_test, p_bal):.4f}, F1: {f1_score(y_test, p_bal):.4f}")
    print(f"3. SMOTE on Train -> Precision: {precision_score(y_test, p_smote):.4f}, Recall: {recall_score(y_test, p_smote):.4f}, F1: {f1_score(y_test, p_smote):.4f}\n")

    # -------------------------------------------------------------
    # TASK 12: HYPERPARAMETER TUNING
    # -------------------------------------------------------------
    print("=========================================================")
    print("      TASK 12: GRIDSEARCHCV & OOB SCORE EVALUATION       ")
    print("=========================================================")

    rf_oob_clf = RandomForestClassifier(oob_score=True, random_state=42)
    tuning_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", rf_oob_clf)])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6, 8],
        "classifier__max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_rf = best_pipeline.named_steps["classifier"]

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best 5-Fold Cross-Validation F1 Score: {grid_search.best_score_:.4f}")
    print(f"Out-of-Bag Score: {best_rf.oob_score_:.4f}\n")

    pipeline_save_path = os.path.join(MODEL_DIR, "full_zepto_titanic_pipeline.pkl")
    joblib.dump(best_pipeline, pipeline_save_path)
    print(f"[SAVED] Complete pipeline artifact saved to: '{pipeline_save_path}'.\n")

    # -------------------------------------------------------------
    # TASK 14: PYTORCH DEEP NEURAL EMBEDDINGS & UNSUPERVISED CLUSTERING
    # -------------------------------------------------------------
    print("=========================================================")
    print("   TASK 14: PYTORCH DEEP EMBEDDINGS & CUSTOMER SEGMENTATION ")
    print("=========================================================")

    # Prepare numeric features for PyTorch Autoencoder training
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num_processed = scaler.fit_transform(num_imputer.fit_transform(df[num_features]))
    
    tensor_x = torch.tensor(X_num_processed, dtype=torch.float32)
    input_dim = X_num_processed.shape[1]
    embedding_dim = 16
    epochs = 50

    autoencoder = CustomerEmbeddingAutoencoder(input_dim=input_dim, embedding_dim=embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)

    autoencoder.train()
    final_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        latent, reconstructed = autoencoder(tensor_x)
        loss = criterion(reconstructed, tensor_x)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    autoencoder.eval()
    with torch.no_grad():
        latent_embeddings, _ = autoencoder(tensor_x)

    # Perform KMeans clustering on PyTorch dense neural embeddings
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(latent_embeddings.numpy())

    # Save PyTorch model and segmented dataset outputs
    pytorch_model_path = os.path.join(MODEL_DIR, "pytorch_customer_autoencoder.pt")
    torch.save(autoencoder.state_dict(), pytorch_model_path)

    segmented_df = df.copy()
    segmented_df["Customer_Cluster"] = cluster_labels
    segmented_csv_path = os.path.join(OUTPUT_DIR, "segmented_customers_output.csv")
    segmented_df.to_csv(segmented_csv_path, index=False)

    pytorch_summary = {
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "final_loss": final_loss,
        "n_clusters": 4
    }
    print(f"[SAVED] PyTorch Autoencoder weights saved to '{pytorch_model_path}'")
    print(f"[SAVED] Segmented customer data saved to '{segmented_csv_path}'\n")

    # -------------------------------------------------------------
    # TASK 13: REGRESSION SIDE-TASK
    # -------------------------------------------------------------
    print("=========================================================")
    print("        TASK 13: REGRESSION SIDE-TASK - PREDICT FARE     ")
    print("=========================================================")

    X_reg = df.drop(columns=["fare", "survived", "alive"])
    y_reg = df["fare"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)

    num_r = ["age", "pclass", "sibsp", "parch"]
    cat_r = ["sex", "embarked"]

    pre_r = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_r),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_r)
    ])

    reg_pipeline = Pipeline([("preprocessor", pre_r), ("regressor", LinearRegression())])
    reg_pipeline.fit(X_train_r, y_train_r)

    y_pred_r = reg_pipeline.predict(X_test_r)

    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)

    n = len(y_test_r)
    p = X_train_r.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"Regression Metrics: MAE = {mae:.2f} | RMSE = {rmse:.2f} | R2 = {r2:.4f} | Adj R2 = {adj_r2:.4f}")

    # Residual Plot
    residuals = y_test_r - y_pred_r
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_r, residuals, alpha=0.6, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residuals")
    plt.title("Fare Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "fare_regression_residuals.png"))
    plt.close()

    reg_metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Adj R2": round(adj_r2, 4)
    }

    # Generate consolidated Markdown Report incorporating all metrics and PyTorch summary
    generate_markdown_report(results, reg_metrics, pytorch_summary)


if __name__ == "__main__":
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "analytics_output.txt")
    
    logger = TeeLogger(log_path, mode="a")
    sys.stdout = logger

    try:
        run_modeling_pipeline()
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"[COMPLETED] Modeling outputs written to {log_path}")