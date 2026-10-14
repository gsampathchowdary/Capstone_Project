import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "full_zepto_titanic_pipeline.pkl")

<<<<<<< HEAD
def verify_pipeline():
    print("=="*30)
    print("      VERIFYING SAVED PIPELINE RELOAD & PREDICTION       ")
    print("=="*30)
=======

def verify_pipeline():
    print("=========================================================")
    print("      VERIFYING SAVED PIPELINE RELOAD & PREDICTION       ")
    print("=========================================================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run 02_modeling.py first.")
>>>>>>> feature/data-pipeline

    # Load complete fitted sklearn pipeline
    pipeline = joblib.load(MODEL_PATH)
    print(f"[SUCCESS] Loaded saved pipeline artifact from '{MODEL_PATH}'.")

    # Pass raw, un-preprocessed new customer records
    raw_sample = pd.DataFrame([
        {
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.33,
            "embarked": "S",
            "class": "First",
            "who": "woman",
            "adult_male": False,
            "embark_town": "Southampton",
            "alone": True
        },
        {
            "pclass": 3,
            "sex": "male",
            "age": 22.0,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked": "S",
            "class": "Third",
            "who": "man",
            "adult_male": True,
            "embark_town": "Southampton",
            "alone": False
        }
    ])

    predictions = pipeline.predict(raw_sample)
    probabilities = pipeline.predict_proba(raw_sample)[:, 1]

    print("\n--- Pipeline Predictions on Raw Unprocessed Input ---")
    for i, pred in enumerate(predictions):
        status = "Survived" if pred == 1 else "Died"
        print(f"Passenger {i+1} ({raw_sample.iloc[i]['sex']}, Class {raw_sample.iloc[i]['pclass']}): Prediction = {status} (Survival Probability = {probabilities[i]:.2%})")

<<<<<<< HEAD
    print("\n✅ PIPELINE VERIFICATION SUCCESSFUL: Pipeline takes raw input end-to-end!")
=======
    # Replaced '\u2705' emoji with clean text tag '[SUCCESS]'
    print("\n[SUCCESS] PIPELINE VERIFICATION SUCCESSFUL: Pipeline takes raw input end-to-end!")

>>>>>>> feature/data-pipeline

if __name__ == "__main__":
    verify_pipeline()