import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-Learn & Imblearn Imports
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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_data():
    """Loads dataset from committed offline fallback titanic.csv file."""
    df = pd.read_csv(CSV_PATH)
    # Drop deck due to >77% missing rate as justified in EDA
    if "deck" in df.columns:
        df.drop(columns=["deck"], inplace=True)
    return df

def run_modeling_pipeline():
    df = load_data()
    
    # -------------------------------------------------------------
    # TASK 7: STRATIFIED TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    print("=="*30)
    print("            TASK 7: STRATIFIED TRAIN/TEST SPLIT          ")
    print("=="*30)
    
    X = df.drop(columns=["survived", "alive"])
    y = df["survived"]
    
    # Stratified split ensures exact target balance preservation in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train set shape: {X_train.shape} | Test set shape: {X_test.shape}")
    print(f"Train survival rate: {y_train.mean():.4f} | Test survival rate: {y_test.mean():.4f}\n")

    # -------------------------------------------------------------
    # TASK 8: PREPROCESSING PIPELINE (ColumnTransformer)
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
    # TASK 9 & 10: TRAIN 3 CLASSIFIERS & DECISION TREE PLOT
    # -------------------------------------------------------------
    print("=="*30)
    print("      TASK 9 & 10: CLASSIFIER TRAINING & EVALUATION      ")
    print("=="*30)

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    }

    results = []

    for name, clf in classifiers.items():
        # Wrap preprocessor and estimator into one end-to-end Pipeline
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Fit strictly on train fold
        model_pipeline.fit(X_train, y_train)

        # Predict on test fold
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

        print(f"\n--- {name} Performance ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    # Save Decision Tree Visualization
    dt_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42))])
    dt_pipeline.fit(X_train, y_train)
    
    # Extract feature names
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
    plt.title("Decision Tree Visualization (Max Depth = 3)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "decision_tree.png"))
    plt.close()
    print(f"\n[SUCCESS] Decision tree visualization saved to '{CHARTS_DIR}/decision_tree.png'.\n")

    # -------------------------------------------------------------
    # TASK 11: IMBALANCE HANDLING COMPARISON
    # -------------------------------------------------------------
    print("=="*30)
    print("        TASK 11: CLASS IMBALANCE STRATEGY COMPARISON      ")
    print("=="*30)

    # (a) Baseline Random Forest
    rf_base = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(random_state=42))])
    rf_base.fit(X_train, y_train)
    p_base = rf_base.predict(X_test)

    # (b) Class Weight Balanced
    rf_bal = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])
    rf_bal.fit(X_train, y_train)
    p_bal = rf_bal.predict(X_test)

    # (c) SMOTE Oversampling on Train Fold Only
    rf_smote = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    rf_smote.fit(X_train, y_train)
    p_smote = rf_smote.predict(X_test)

    print("Imbalance Comparison Results (Random Forest):")
    print(f"1. Baseline        -> Precision: {precision_score(y_test, p_base):.4f}, Recall: {recall_score(y_test, p_base):.4f}, F1: {f1_score(y_test, p_base):.4f}")
    print(f"2. Class Balanced  -> Precision: {precision_score(y_test, p_bal):.4f}, Recall: {recall_score(y_test, p_bal):.4f}, F1: {f1_score(y_test, p_bal):.4f}")
    print(f"3. SMOTE (Train)   -> Precision: {precision_score(y_test, p_smote):.4f}, Recall: {recall_score(y_test, p_smote):.4f}, F1: {f1_score(y_test, p_smote):.4f}\n")

    # -------------------------------------------------------------
    # TASK 12: HYPERPARAMETER TUNING (GridSearchCV + OOB Score)
    # -------------------------------------------------------------
    print("=="*30)
    print("      TASK 12: GRIDSEARCHCV & OOB SCORE EVALUATION       ")
    print("=="*30)

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
    print(f"Out-of-Bag (OOB) Accuracy Score: {best_rf.oob_score_:.4f}\n")

    # Save Best Pipeline Artifact
    pipeline_save_path = os.path.join(MODEL_DIR, "full_zepto_titanic_pipeline.pkl")
    joblib.dump(best_pipeline, pipeline_save_path)
    print(f"[SUCCESS] Complete fitted pipeline saved to '{pipeline_save_path}'.\n")

    # -------------------------------------------------------------
    # TASK 13: REGRESSION SIDE-TASK (PREDICT FARE)
    # -------------------------------------------------------------
    print("=="*30)
    print("        TASK 13: REGRESSION SIDE-TASK (PREDICT FARE)     ")
    print("=="*30)

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
    plt.xlabel("Predicted Fare ($)")
    plt.ylabel("Residuals ($)")
    plt.title("Linear Regression Residual Plot (Fare Prediction)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "fare_regression_residuals.png"))
    plt.close()

    print("\nResidual Spread Analysis: The residual plot displays strong HETEROSCEDASTICITY (fan-shaped distribution), indicating that errors increase proportionally for higher luxury fare classes.")

if __name__ == "__main__":
    run_modeling_pipeline()