"""
Stage 1: Data Pipeline & Model Training
Customer Churn and Behaviour Prediction Project

Main execution script:
1. Generates or loads the Telco Customer Churn dataset
2. Fits preprocessing pipeline (imputation, scaling, one-hot encoding)
3. Trains and benchmarks 3 models: Logistic Regression, Random Forest, XGBoost
4. Evaluates ROC-AUC, F1-score, Precision, Recall, Confusion Matrix
5. Selects and persists the best model and preprocessor (joblib)
6. Computes global and local SHAP feature importance for risk interpretation
7. Demonstrates live inference & explainability on test customer profiles
"""

import os
import json
import pandas as pd
import numpy as np

from src.data_generator import load_or_generate_dataset
from src.pipeline import (
    load_pipeline,
    prepare_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from src.train import train_and_evaluate
from src.explainability import ChurnExplainer, save_explainer


def run_pipeline():
    print("=" * 80)
    print(" STAGE 1: TELCO CHURN PREDICTION PIPELINE & MODEL BENCHMARKING")
    print("=" * 80)

    # 1. Train and evaluate all models, pick best, and save artifacts
    best_model, preprocessor, results, X_test_proc, y_test, feature_names = train_and_evaluate(
        data_path=None,
        artifacts_dir="models",
        test_size=0.20,
        random_state=42
    )

    # 2. Build SHAP Explainer
    print("\n" + "=" * 80)
    print("                   SHAP FEATURE IMPORTANCE & EXPLAINABILITY")
    print("=" * 80)
    
    explainer = ChurnExplainer(
        model=best_model,
        feature_names=feature_names,
        background_data=X_test_proc[:200]
    )

    # Save explainer artifact
    save_explainer(explainer, "models/shap_explainer.joblib")

    # Global feature importance
    print("\n[SHAP] Computing global feature importance across test population...")
    global_importance = explainer.get_global_importance(X_test_proc[:300])
    print("\nTop 10 Global Features Driving Churn Decisions (Mean |SHAP|):")
    print(global_importance.head(10).to_string(index=False))

    # 3. Demonstrate Local SHAP Explanations on Diverse Customer Profiles
    print("\n" + "=" * 80)
    print("         SAMPLE INFERENCE & INDIVIDUAL CUSTOMER RISK EXPLANATIONS")
    print("=" * 80)

    # Sample profile A: High risk (short tenure, month-to-month, many tickets, high charges, inactive)
    high_risk_customer = {
        "customer_id": "CUST-99001",
        "tenure": 2,
        "monthly_charges": 98.50,
        "total_charges": 197.00,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "support_tickets": 5,
        "last_login_days_ago": 48,
        "usage_frequency": 12.0
    }

    # Sample profile B: Low risk (long tenure, two-year contract, bank transfer, low tickets, active)
    low_risk_customer = {
        "customer_id": "CUST-99002",
        "tenure": 54,
        "monthly_charges": 45.00,
        "total_charges": 2430.00,
        "contract_type": "Two year",
        "payment_method": "Bank transfer (automatic)",
        "internet_service": "DSL",
        "support_tickets": 0,
        "last_login_days_ago": 3,
        "usage_frequency": 78.5
    }

    for cust in [high_risk_customer, low_risk_customer]:
        # Preprocess single customer
        cust_df = pd.DataFrame([cust])
        cust_proc = preprocessor.transform(cust_df)
        
        explanation = explainer.explain_instance(
            customer_processed_row=cust_proc,
            customer_raw_data=cust,
            top_k=3
        )

        print(f"\n--- Customer Explanation: {explanation['customer_id']} ---")
        print(f"Risk Tier          : {explanation['risk_tier']}")
        print(f"Churn Probability  : {explanation['churn_probability']:.1%}")
        print(f"Predicted Action   : {'CHURN RISK (1)' if explanation['predicted_churn'] == 1 else 'RETAINED (0)'}")
        print("\n  Top Factors Increasing Churn Risk:")
        for factor in explanation['top_risk_factors']:
            print(f"    [+] {factor['feature']:<30} (SHAP impact: +{factor['shap_value']:.4f})")
            
        print("\n  Top Factors Protecting Against Churn:")
        for factor in explanation['top_protective_factors']:
            print(f"    [-] {factor['feature']:<30} (SHAP impact: {factor['shap_value']:.4f})")

    # 4. Summary of Saved Artifacts
    print("\n" + "=" * 80)
    print("                      SAVED PRODUCTION ARTIFACTS")
    print("=" * 80)
    saved_files = [
        "models/preprocessor.joblib",
        "models/best_model.joblib",
        "models/full_pipeline.joblib",
        "models/shap_explainer.joblib",
        "models/model_metadata.json",
        "data/telco_customer_churn_synthetic.csv"
    ]
    for sf in saved_files:
        if os.path.exists(sf):
            size_kb = os.path.getsize(sf) / 1024
            print(f"  [OK] {sf:<45} ({size_kb:>8.2f} KB)")

    print("\n[Stage 1 Completed Successfully!]")


if __name__ == "__main__":
    run_pipeline()
