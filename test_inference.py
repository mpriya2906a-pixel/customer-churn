"""
Inference & API loading demonstration script.
Shows how to load saved joblib artifacts and generate churn predictions + SHAP explanations.
"""

import json
import joblib
import pandas as pd
from src.explainability import ChurnExplainer, load_explainer


def run_standalone_inference():
    print("Loading serialized artifacts from models/ directory...")
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/best_model.joblib")
    explainer: ChurnExplainer = load_explainer("models/shap_explainer.joblib")

    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    print(f"Loaded Best Model: {metadata['best_model_name']}")

    # Example incoming API payload
    customer_payload = {
        "customer_id": "API-TEST-001",
        "tenure": 4,
        "monthly_charges": 89.20,
        "total_charges": 356.80,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "support_tickets": 3,
        "last_login_days_ago": 25,
        "usage_frequency": 22.0
    }

    df_single = pd.DataFrame([customer_payload])
    X_processed = preprocessor.transform(df_single)

    # Prediction
    prob = model.predict_proba(X_processed)[0, 1]
    prediction = int(prob >= 0.5)

    # Explanation
    explanation = explainer.explain_instance(
        customer_processed_row=X_processed,
        customer_raw_data=customer_payload,
        top_k=3
    )

    print("\nAPI Response Output:")
    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    run_standalone_inference()
